## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 886.64361740241


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742)
1: (-437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521)
2: (-439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498)
3: (-536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455)
4: (-473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094)

## BASE Result
execution time: IAR + LP analysis = 1.57 + 2.04 = 3.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -886.6916286, upper bound: 886.6916286


# Binary Search by BASE starts (time budget: 1196.39 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=1024.72607421875
rel_dist={0: [-886.6916286331143, 886.6916286331141]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=1024.72607421875
rel_dist={0: [-886.6909129349328, 886.6909129349328]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=1024.72607421875
rel_dist={0: [-886.6885385920837, 886.6885385920837]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=1024.72607421875
rel_dist={0: [-886.6867135844589, 886.6867135844589]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=1024.72607421875
rel_dist={0: [-886.6855568160574, 886.6855568160574]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=1024.72607421875
rel_dist={0: [-886.6846760554688, 886.6846760554686]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=1024.72607421875
rel_dist={0: [-886.6841389728621, 886.684138972862]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=1024.72607421875
rel_dist={0: [-886.6838666122214, 886.6838666122214]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=1024.72607421875
rel_dist={0: [-886.6837304319017, 886.6837304319015]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=1024.72607421875
rel_dist={0: [-886.6836623417435, 886.6836623417435]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=1024.72607421875
rel_dist={0: [-886.6836282966669, 886.6836282966672]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=1024.72607421875
rel_dist={0: [-886.6836112741353, 886.683611274135]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=1024.72607421875
rel_dist={0: [-886.6836027628814, 886.683602762881]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=1024.72607421875
rel_dist={0: [-886.6835985072787, 886.6835985072787]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=1024.72607421875
rel_dist={0: [-886.6835963802994, 886.6835963795252]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=1024.72607421875
rel_dist={0: [-886.6835953040254, 886.6835953042985]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=1024.72607421875
rel_dist={0: [-886.6835947653071, 886.6835947653071]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=1024.72607421875
rel_dist={0: [-886.6835944967892, 886.6835945036696]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=1024.72607421875
rel_dist={0: [-886.6835943626116, 886.6835943634572]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=1024.72607421875
rel_dist={0: [-886.6835943127753, 886.6835943184742]}

## Binary Search Result
Binary search time: 76.45 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1119.94 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6916205, upper bound: 886.6916205
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6916205, upper bound: 886.6916205
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 0, lower bound: -886.6916205, upper bound: 886.6916205
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 0, lower bound: -886.6916205, upper bound: 886.6916205

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6886522, upper bound: 886.6897593
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6886522, upper bound: 886.6886522
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6899717, upper bound: 886.6899717
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6899717, upper bound: 886.6899717
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.10 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 0, lower bound: -886.6886522, upper bound: 886.6897593
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 0, lower bound: -886.6886522, upper bound: 886.6886522
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 0, lower bound: -886.6899717, upper bound: 886.6899717
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 0, lower bound: -886.6899717, upper bound: 886.6899717

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841925, upper bound: 886.6841043
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841925, upper bound: 886.6841043
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6892514, upper bound: 886.6873990
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6874312
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6896413, upper bound: 886.6873633
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6873633, upper bound: 886.6896413
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6267795, upper bound: 886.6248161
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6267795, upper bound: 886.6254175
time: 0.88 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -886.6841925, upper bound: 886.6841043
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -886.6841925, upper bound: 886.6841043
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -886.6892514, upper bound: 886.6873990
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6874312
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -886.6896413, upper bound: 886.6873633
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.29
Output dim: 0, lower bound: -886.6873633, upper bound: 886.6896413
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.29
Output dim: 0, lower bound: -886.6267795, upper bound: 886.6248161
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.29
Output dim: 0, lower bound: -886.6267795, upper bound: 886.6254175

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841925, upper bound: 886.6841043
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841925, upper bound: 886.6841043
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6863477
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6881054, upper bound: 886.6860917
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6874312
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6870944
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6267327, upper bound: 886.6246164
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6267327, upper bound: 886.6246164
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6246164, upper bound: 886.6247082
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6246164, upper bound: 886.6253824
time: 0.94 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6841925, upper bound: 886.6841043
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6841925, upper bound: 886.6841043
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6863477
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6881054, upper bound: 886.6860917
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6874312
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6870944
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6267327, upper bound: 886.6246164
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6267327, upper bound: 886.6246164
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6246164, upper bound: 886.6247082
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.44
Output dim: 0, lower bound: -886.6246164, upper bound: 886.6253824

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6840397, upper bound: 886.6840397
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841278, upper bound: 886.6840397
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6840397, upper bound: 886.6840397
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841278, upper bound: 886.6840397
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6846621
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6242284, upper bound: 886.6242284
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6242284, upper bound: 886.6242284
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6813011
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6814711
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6870944
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6870944
time: 0.83 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6840397, upper bound: 886.6840397
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6841278, upper bound: 886.6840397
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6840397, upper bound: 886.6840397
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6841278, upper bound: 886.6840397
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6846621
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6242284, upper bound: 886.6242284
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6242284, upper bound: 886.6242284
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6813011
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6814711
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6870944
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.04
Output dim: 0, lower bound: -886.6870944, upper bound: 886.6870944

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6828866, upper bound: 886.6828866
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6829631, upper bound: 886.6828866
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841278, upper bound: 886.6840397
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6840698, upper bound: 886.6840397
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6846340
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800111
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800111
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6813011
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6811365
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6811960
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6860917
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6860917
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6860917
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6860917
time: 0.79 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6828866, upper bound: 886.6828866
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6829631, upper bound: 886.6828866
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6841278, upper bound: 886.6840397
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6840698, upper bound: 886.6840397
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6841043, upper bound: 886.6841043
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6846340
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800111
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800111
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6813011
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6811365
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6811960
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6860917
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6860917
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6860917
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 0, lower bound: -886.6860917, upper bound: 886.6860917

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6829631, upper bound: 886.6828866
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6828866, upper bound: 886.6828866
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802993, upper bound: 886.6802993
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802993, upper bound: 886.6802993
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808803, upper bound: 886.6808803
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808803, upper bound: 886.6808803
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841278, upper bound: 886.6840396
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6823614, upper bound: 886.6821580
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6821580
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800545
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6802475
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6783764, upper bound: 886.6783764
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6783764, upper bound: 886.6783764
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6726242, upper bound: 886.6726242
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6726242, upper bound: 886.6726242
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6812984
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6811365
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6803865, upper bound: 886.6803865
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6803865, upper bound: 886.6803865
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6811960
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6845761, upper bound: 886.6845761
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6845761, upper bound: 886.6845761
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 0.92 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6829631, upper bound: 886.6828866
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6828866, upper bound: 886.6828866
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6802993, upper bound: 886.6802993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6802993, upper bound: 886.6802993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808803, upper bound: 886.6808803
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808803, upper bound: 886.6808803
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6841278, upper bound: 886.6840396
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6823614, upper bound: 886.6821580
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6821580, upper bound: 886.6821580
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6840396, upper bound: 886.6840396
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800545
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6802475
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6783764, upper bound: 886.6783764
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6783764, upper bound: 886.6783764
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6726242, upper bound: 886.6726242
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6726242, upper bound: 886.6726242
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6812984
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6811365, upper bound: 886.6811365
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6803865, upper bound: 886.6803865
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6803865, upper bound: 886.6803865
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6811960
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6845761, upper bound: 886.6845761
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6845761, upper bound: 886.6845761
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.78
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6678960, upper bound: 886.6678960
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6678960, upper bound: 886.6678960
time: 1.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693599, upper bound: 886.6693599
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6693599, upper bound: 886.6693599
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6803951, upper bound: 886.6803951
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6804484, upper bound: 886.6803951
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800345, upper bound: 886.6800345
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800345, upper bound: 886.6800345
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6828866, upper bound: 886.6828866
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6828866, upper bound: 886.6828866
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802432, upper bound: 886.6802432
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802432, upper bound: 886.6802432
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802993, upper bound: 886.6802993
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802993, upper bound: 886.6802993
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6714850, upper bound: 886.6714850
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6714850, upper bound: 886.6714850
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813993, upper bound: 886.6813993
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813993, upper bound: 886.6813993
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816438, upper bound: 886.6813993
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816438, upper bound: 886.6813993
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6799473, upper bound: 886.6799473
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6799473, upper bound: 886.6799473
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6679680, upper bound: 886.6679680
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6679680, upper bound: 886.6679680
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802432, upper bound: 886.6802432
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802432, upper bound: 886.6802432
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6821541, upper bound: 886.6821541
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6821541, upper bound: 886.6821541
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6833785, upper bound: 886.6832704
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813993, upper bound: 886.6813993
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6816154, upper bound: 886.6813993
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6810540, upper bound: 886.6810540
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6812478, upper bound: 886.6810540
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808803, upper bound: 886.6808803
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808803, upper bound: 886.6808803
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822359, upper bound: 886.6822359
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6822359, upper bound: 886.6822359
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6696921, upper bound: 886.6696921
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6696921, upper bound: 886.6696921
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6679680, upper bound: 886.6679680
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6679680, upper bound: 886.6679680
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6628062, upper bound: 886.6628062
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6628062, upper bound: 886.6628062
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6802475
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800929
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841772, upper bound: 886.6841772
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841772, upper bound: 886.6841772
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6783742, upper bound: 886.6783742
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6783742, upper bound: 886.6783742
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710924, upper bound: 886.6710924
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710924, upper bound: 886.6710924
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6621272, upper bound: 886.6621272
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6621272, upper bound: 886.6621272
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6723851, upper bound: 886.6723851
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6723851, upper bound: 886.6723851
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6810224
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6743207, upper bound: 886.6743207
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6743207, upper bound: 886.6743207
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6740839, upper bound: 886.6740839
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6740839, upper bound: 886.6740839
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6803865, upper bound: 886.6803865
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6803865, upper bound: 886.6803865
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6803088
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800111
time: 0.90 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6678960, upper bound: 886.6678960
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6678960, upper bound: 886.6678960
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6693599, upper bound: 886.6693599
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6693599, upper bound: 886.6693599
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6803951, upper bound: 886.6803951
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6804484, upper bound: 886.6803951
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6800345, upper bound: 886.6800345
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6800345, upper bound: 886.6800345
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6828866, upper bound: 886.6828866
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6828866, upper bound: 886.6828866
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6802432, upper bound: 886.6802432
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6802432, upper bound: 886.6802432
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6802993, upper bound: 886.6802993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6802993, upper bound: 886.6802993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6714850, upper bound: 886.6714850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6714850, upper bound: 886.6714850
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6813993, upper bound: 886.6813993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6813993, upper bound: 886.6813993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6816438, upper bound: 886.6813993
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6816438, upper bound: 886.6813993
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6799473, upper bound: 886.6799473
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6799473, upper bound: 886.6799473
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6679680, upper bound: 886.6679680
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6679680, upper bound: 886.6679680
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6802432, upper bound: 886.6802432
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6802432, upper bound: 886.6802432
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822272, upper bound: 886.6822272
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6821541, upper bound: 886.6821541
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6821541, upper bound: 886.6821541
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6833785, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6813993, upper bound: 886.6813993
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6816154, upper bound: 886.6813993
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6810540, upper bound: 886.6810540
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6812478, upper bound: 886.6810540
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6808803, upper bound: 886.6808803
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6808803, upper bound: 886.6808803
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6718587, upper bound: 886.6718587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822359, upper bound: 886.6822359
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6822359, upper bound: 886.6822359
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6832704, upper bound: 886.6832704
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6696921, upper bound: 886.6696921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6696921, upper bound: 886.6696921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6679680, upper bound: 886.6679680
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6679680, upper bound: 886.6679680
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716849, upper bound: 886.6716849
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6716456, upper bound: 886.6716456
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6628062, upper bound: 886.6628062
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6628062, upper bound: 886.6628062
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6802475
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800929
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6841772, upper bound: 886.6841772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6841772, upper bound: 886.6841772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6844119, upper bound: 886.6844119
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6783742, upper bound: 886.6783742
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6783742, upper bound: 886.6783742
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6710924, upper bound: 886.6710924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6710924, upper bound: 886.6710924
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6621272, upper bound: 886.6621272
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6621272, upper bound: 886.6621272
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6723851, upper bound: 886.6723851
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6723851, upper bound: 886.6723851
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6810224
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6743207, upper bound: 886.6743207
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6743207, upper bound: 886.6743207
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6740839, upper bound: 886.6740839
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6740839, upper bound: 886.6740839
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6803865, upper bound: 886.6803865
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6803865, upper bound: 886.6803865
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6803088
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.69
Output dim: 0, lower bound: -886.6800111, upper bound: 886.6800111
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6845761, upper bound: 886.6845761
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6845761, upper bound: 886.6845761
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.69
Output dim: 0, lower bound: -886.6808968, upper bound: 886.6808968
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=1024.72607421875
rel_dist={0: [-886.6916286331143, 886.6916286331141]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6894565, upper bound: 886.6898408
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6898408, upper bound: 886.6894565
time: 1.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.18 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -886.6894565, upper bound: 886.6898408
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.18
Output dim: 0, lower bound: -886.6898408, upper bound: 886.6894565

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6860291, upper bound: 886.6857335
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6860291, upper bound: 886.6855725
time: 0.86 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6855725, upper bound: 886.6860291
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6857335, upper bound: 886.6860291
time: 1.04 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.66 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 0, lower bound: -886.6860291, upper bound: 886.6857335
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 0, lower bound: -886.6860291, upper bound: 886.6855725
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 0, lower bound: -886.6855725, upper bound: 886.6860291
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 0, lower bound: -886.6857335, upper bound: 886.6860291

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6853689, upper bound: 886.6856626
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6855725, upper bound: 886.6857335
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6857377, upper bound: 886.6854890
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6857377, upper bound: 886.6853497
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6833012, upper bound: 886.6834893
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6832593, upper bound: 886.6834893
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844720, upper bound: 886.6849389
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6846721, upper bound: 886.6844689
time: 1.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -886.6853689, upper bound: 886.6856626
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -886.6855725, upper bound: 886.6857335
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -886.6857377, upper bound: 886.6854890
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -886.6857377, upper bound: 886.6853497
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -886.6833012, upper bound: 886.6834893
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -886.6832593, upper bound: 886.6834893
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -886.6844720, upper bound: 886.6849389
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -886.6846721, upper bound: 886.6844689

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6742766, upper bound: 886.6742765
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6742766, upper bound: 886.6742766
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6742766, upper bound: 886.6742766
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6742766, upper bound: 886.6742765
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6749670, upper bound: 886.6749670
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6749670, upper bound: 886.6749670
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6857377, upper bound: 886.6853497
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6856465, upper bound: 886.6853497
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6817499, upper bound: 886.6813799
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6817499, upper bound: 886.6816261
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6712779, upper bound: 886.6712779
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6712779, upper bound: 886.6714642
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6842939, upper bound: 886.6845985
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844720, upper bound: 886.6849389
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6844397, upper bound: 886.6842969
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6843661, upper bound: 886.6841201
time: 0.94 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.49 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6742766, upper bound: 886.6742765
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6742766, upper bound: 886.6742766
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6742766, upper bound: 886.6742766
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6742766, upper bound: 886.6742765
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6749670, upper bound: 886.6749670
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6749670, upper bound: 886.6749670
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6857377, upper bound: 886.6853497
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6856465, upper bound: 886.6853497
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6817499, upper bound: 886.6813799
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6817499, upper bound: 886.6816261
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6712779, upper bound: 886.6712779
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6712779, upper bound: 886.6714642
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6842939, upper bound: 886.6845985
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6844720, upper bound: 886.6849389
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6844397, upper bound: 886.6842969
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.49
Output dim: 0, lower bound: -886.6843661, upper bound: 886.6841201

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6522210, upper bound: 886.6540109
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6522210, upper bound: 886.6524402
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6429414, upper bound: 886.6436513
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6429414, upper bound: 886.6436513
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6741723, upper bound: 886.6741723
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6741723, upper bound: 886.6741723
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6665367, upper bound: 886.6665367
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6665710, upper bound: 886.6665367
time: 1.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6741335, upper bound: 886.6741335
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6741335, upper bound: 886.6741335
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6749106, upper bound: 886.6749387
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6749106, upper bound: 886.6749391
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6826374, upper bound: 886.6823348
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6823348, upper bound: 886.6823348
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813783, upper bound: 886.6813783
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6817483, upper bound: 886.6813783
time: 3.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797485, upper bound: 886.6797485
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797485, upper bound: 886.6797485
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6712654
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6711094
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6815845, upper bound: 886.6815845
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6815845, upper bound: 886.6817933
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6755872
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6755872
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6838823, upper bound: 886.6840177
time: 1.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6841957, upper bound: 886.6840638
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5930997, upper bound: 886.5933723
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5930997, upper bound: 886.5933723
time: 0.87 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6522210, upper bound: 886.6540109
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6522210, upper bound: 886.6524402
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6429414, upper bound: 886.6436513
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6429414, upper bound: 886.6436513
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6741723, upper bound: 886.6741723
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6741723, upper bound: 886.6741723
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6665367, upper bound: 886.6665367
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6665710, upper bound: 886.6665367
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6741335, upper bound: 886.6741335
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6741335, upper bound: 886.6741335
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6749106, upper bound: 886.6749387
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6749106, upper bound: 886.6749391
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6826374, upper bound: 886.6823348
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6823348, upper bound: 886.6823348
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6813783, upper bound: 886.6813783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6817483, upper bound: 886.6813783
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6797485, upper bound: 886.6797485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6797485, upper bound: 886.6797485
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6712654
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6711094
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6815845, upper bound: 886.6815845
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6815845, upper bound: 886.6817933
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6755872
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6755872
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6838823, upper bound: 886.6840177
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.6841957, upper bound: 886.6840638
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.5930997, upper bound: 886.5933723
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.35
Output dim: 0, lower bound: -886.5930997, upper bound: 886.5933723

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6489922, upper bound: 886.6505329
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6489922, upper bound: 886.6504491
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6511407, upper bound: 886.6511407
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6511407, upper bound: 886.6513189
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6427489, upper bound: 886.6434526
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6427489, upper bound: 886.6433238
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6429414, upper bound: 886.6436513
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6429414, upper bound: 886.6433798
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6740009, upper bound: 886.6740009
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6740009, upper bound: 886.6740009
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731789, upper bound: 886.6731789
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731789, upper bound: 886.6731789
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6664438, upper bound: 886.6663805
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6663805, upper bound: 886.6663805
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6664438, upper bound: 886.6663805
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6664437, upper bound: 886.6663805
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737527, upper bound: 886.6737527
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6737527, upper bound: 886.6737527
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731288, upper bound: 886.6731288
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731288, upper bound: 886.6731288
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6662018, upper bound: 886.6662591
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6662018, upper bound: 886.6662018
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6720585, upper bound: 886.6720705
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6720585, upper bound: 886.6720585
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6776425, upper bound: 886.6776425
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6776425, upper bound: 886.6776425
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814988, upper bound: 886.6814988
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6814988, upper bound: 886.6814988
time: 1.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706016, upper bound: 886.6705885
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706112, upper bound: 886.6705885
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813783, upper bound: 886.6813783
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813783, upper bound: 886.6813783
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6788031, upper bound: 886.6788031
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6788031, upper bound: 886.6788031
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797485, upper bound: 886.6797485
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797485, upper bound: 886.6797485
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6690880, upper bound: 886.6687482
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6690880, upper bound: 886.6687482
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6712654
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6689279, upper bound: 886.6687482
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6689279, upper bound: 886.6687482
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6365919, upper bound: 886.6365919
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6365919, upper bound: 886.6365919
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6815845, upper bound: 886.6817933
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6815845, upper bound: 886.6816815
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6739955, upper bound: 886.6741448
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6739955, upper bound: 886.6739955
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6755872
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6754238
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6826785, upper bound: 886.6826785
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6826785, upper bound: 886.6826875
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6780418, upper bound: 886.6764508
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6780418, upper bound: 886.6764508
time: 0.91 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6489922, upper bound: 886.6505329
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6489922, upper bound: 886.6504491
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6511407, upper bound: 886.6511407
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6511407, upper bound: 886.6513189
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6427489, upper bound: 886.6434526
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6427489, upper bound: 886.6433238
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6429414, upper bound: 886.6436513
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6429414, upper bound: 886.6433798
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6740009, upper bound: 886.6740009
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6740009, upper bound: 886.6740009
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6731789, upper bound: 886.6731789
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6731789, upper bound: 886.6731789
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6664438, upper bound: 886.6663805
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6663805, upper bound: 886.6663805
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6664438, upper bound: 886.6663805
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6664437, upper bound: 886.6663805
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6737527, upper bound: 886.6737527
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6737527, upper bound: 886.6737527
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6731288, upper bound: 886.6731288
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6731288, upper bound: 886.6731288
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6662018, upper bound: 886.6662591
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6662018, upper bound: 886.6662018
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6720585, upper bound: 886.6720705
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6720585, upper bound: 886.6720585
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6776425, upper bound: 886.6776425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6776425, upper bound: 886.6776425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6814988, upper bound: 886.6814988
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6814988, upper bound: 886.6814988
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6706016, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6706112, upper bound: 886.6705885
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6813783, upper bound: 886.6813783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6813783, upper bound: 886.6813783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6788031, upper bound: 886.6788031
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6788031, upper bound: 886.6788031
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6797485, upper bound: 886.6797485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6797485, upper bound: 886.6797485
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6690880, upper bound: 886.6687482
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6690880, upper bound: 886.6687482
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6712654
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6689279, upper bound: 886.6687482
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6689279, upper bound: 886.6687482
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6365919, upper bound: 886.6365919
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6365919, upper bound: 886.6365919
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6815845, upper bound: 886.6817933
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6815845, upper bound: 886.6816815
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6739955, upper bound: 886.6741448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6739955, upper bound: 886.6739955
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6755872
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6754238
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6826785, upper bound: 886.6826785
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6826785, upper bound: 886.6826875
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6780418, upper bound: 886.6764508
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 0, lower bound: -886.6780418, upper bound: 886.6764508

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488708, upper bound: 886.6503763
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6488708, upper bound: 886.6488708
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6424940, upper bound: 886.6424940
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6424940, upper bound: 886.6433506
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6508672, upper bound: 886.6508672
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6508672, upper bound: 886.6508672
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505762, upper bound: 886.6508655
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6505762, upper bound: 886.6508817
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6427071, upper bound: 886.6434005
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6427071, upper bound: 886.6429788
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5911232, upper bound: 886.5911232
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5911232, upper bound: 886.5911232
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6403754, upper bound: 886.6403754
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6403754, upper bound: 886.6403754
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718633, upper bound: 886.6719179
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6718633, upper bound: 886.6718633
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731068, upper bound: 886.6731068
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731068, upper bound: 886.6731068
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6663068, upper bound: 886.6663068
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6663068, upper bound: 886.6663068
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6659395, upper bound: 886.6658461
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6658461, upper bound: 886.6658461
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6592562, upper bound: 886.6592562
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6592562, upper bound: 886.6592562
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6653129, upper bound: 886.6653129
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6653129, upper bound: 886.6653129
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6736338, upper bound: 886.6736338
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6736338, upper bound: 886.6736338
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6427071, upper bound: 886.6427071
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6427071, upper bound: 886.6427071
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731288, upper bound: 886.6731288
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6731288, upper bound: 886.6731288
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6722347, upper bound: 886.6722347
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6722347, upper bound: 886.6722347
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6660475, upper bound: 886.6661060
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6660475, upper bound: 886.6660475
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6661240, upper bound: 886.6661240
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6661240, upper bound: 886.6661240
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6705814, upper bound: 886.6705814
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6705814, upper bound: 886.6706156
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6720585, upper bound: 886.6720585
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6720585, upper bound: 886.6720585
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6746192, upper bound: 886.6746192
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6746192, upper bound: 886.6746192
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6776425, upper bound: 886.6776425
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6776425, upper bound: 886.6776425
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811564, upper bound: 886.6811564
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6811564, upper bound: 886.6811564
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6735290, upper bound: 886.6735290
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6735290, upper bound: 886.6735290
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6703912, upper bound: 886.6703912
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6703912, upper bound: 886.6703912
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6705885, upper bound: 886.6705885
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6705885, upper bound: 886.6705885
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6706112, upper bound: 886.6705885
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6705885, upper bound: 886.6705885
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813245, upper bound: 886.6813245
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6813245, upper bound: 886.6813245
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6669369, upper bound: 886.6669369
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6669369, upper bound: 886.6669369
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6787849, upper bound: 886.6787849
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6787849, upper bound: 886.6787849
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6779112, upper bound: 886.6779112
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6779112, upper bound: 886.6779112
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797261, upper bound: 886.6797261
time: 1.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6797261, upper bound: 886.6797261
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
time: 1.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
time: 1.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6687482, upper bound: 886.6687482
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6687482, upper bound: 886.6687482
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6664079, upper bound: 886.6664079
time: 1.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6664079, upper bound: 886.6664079
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6664079, upper bound: 886.6664079
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6664079, upper bound: 886.6664079
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6709067, upper bound: 886.6709067
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6709067, upper bound: 886.6709067
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6712654
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6687482, upper bound: 886.6687482
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6689279, upper bound: 886.6687482
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6687482, upper bound: 886.6687482
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6689279, upper bound: 886.6687482
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6758179, upper bound: 886.6760346
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6758179, upper bound: 886.6760698
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802835, upper bound: 886.6803959
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6802835, upper bound: 886.6803292
time: 1.15 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6488708, upper bound: 886.6503763
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6488708, upper bound: 886.6488708
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6424940, upper bound: 886.6424940
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6424940, upper bound: 886.6433506
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6508672, upper bound: 886.6508672
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6508672, upper bound: 886.6508672
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6505762, upper bound: 886.6508655
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6505762, upper bound: 886.6508817
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6427071, upper bound: 886.6434005
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6427071, upper bound: 886.6429788
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.5911232, upper bound: 886.5911232
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.5911232, upper bound: 886.5911232
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6403754, upper bound: 886.6403754
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6403754, upper bound: 886.6403754
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6718633, upper bound: 886.6719179
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6718633, upper bound: 886.6718633
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6731068, upper bound: 886.6731068
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6731068, upper bound: 886.6731068
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6663068, upper bound: 886.6663068
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6663068, upper bound: 886.6663068
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6659395, upper bound: 886.6658461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6658461, upper bound: 886.6658461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6592562, upper bound: 886.6592562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6592562, upper bound: 886.6592562
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6653129, upper bound: 886.6653129
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6653129, upper bound: 886.6653129
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6736338, upper bound: 886.6736338
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6736338, upper bound: 886.6736338
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6427071, upper bound: 886.6427071
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6427071, upper bound: 886.6427071
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6731288, upper bound: 886.6731288
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6731288, upper bound: 886.6731288
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6722347, upper bound: 886.6722347
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6722347, upper bound: 886.6722347
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6660475, upper bound: 886.6661060
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6660475, upper bound: 886.6660475
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6661240, upper bound: 886.6661240
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6661240, upper bound: 886.6661240
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6705814, upper bound: 886.6705814
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6705814, upper bound: 886.6706156
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6720585, upper bound: 886.6720585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6720585, upper bound: 886.6720585
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6746192, upper bound: 886.6746192
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6746192, upper bound: 886.6746192
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6776425, upper bound: 886.6776425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6776425, upper bound: 886.6776425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6811564, upper bound: 886.6811564
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6811564, upper bound: 886.6811564
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6735290, upper bound: 886.6735290
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6735290, upper bound: 886.6735290
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6703912, upper bound: 886.6703912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6703912, upper bound: 886.6703912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6705885, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6705885, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6706163, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6706112, upper bound: 886.6705885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6705885, upper bound: 886.6705885
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6813245, upper bound: 886.6813245
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6813245, upper bound: 886.6813245
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6669369, upper bound: 886.6669369
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6669369, upper bound: 886.6669369
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6787849, upper bound: 886.6787849
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6787849, upper bound: 886.6787849
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6779112, upper bound: 886.6779112
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6779112, upper bound: 886.6779112
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6797261, upper bound: 886.6797261
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6797261, upper bound: 886.6797261
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6796718, upper bound: 886.6796718
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6687482, upper bound: 886.6687482
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6687482, upper bound: 886.6687482
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6664079, upper bound: 886.6664079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6664079, upper bound: 886.6664079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6664079, upper bound: 886.6664079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6664079, upper bound: 886.6664079
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6709067, upper bound: 886.6709067
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6709067, upper bound: 886.6709067
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6710769
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6710769, upper bound: 886.6712654
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6687482, upper bound: 886.6687482
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6689279, upper bound: 886.6687482
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6687482, upper bound: 886.6687482
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6689279, upper bound: 886.6687482
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6758179, upper bound: 886.6760346
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6758179, upper bound: 886.6760698
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6802835, upper bound: 886.6803959
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -886.6802835, upper bound: 886.6803292
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -886.6739955, upper bound: 886.6741448
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -886.6739955, upper bound: 886.6739955
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6755872
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -886.6754238, upper bound: 886.6754238
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -886.6826785, upper bound: 886.6826785
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -886.6826785, upper bound: 886.6826875
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -886.6780418, upper bound: 886.6764508
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.10
Output dim: 0, lower bound: -886.6780418, upper bound: 886.6764508
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=1024.72607421875
rel_dist={0: [-886.6909129349328, 886.6909129349328]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6769298, upper bound: 886.6769298
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6769298, upper bound: 886.6769298
time: 0.95 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 0, lower bound: -886.6769298, upper bound: 886.6769298
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 0, lower bound: -886.6769298, upper bound: 886.6769298

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760628, upper bound: 886.6769298
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6760628, upper bound: 886.6759842
time: 1.21 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765041, upper bound: 886.6765041
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6765041, upper bound: 886.6765041
time: 0.92 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -886.6760628, upper bound: 886.6769298
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -886.6760628, upper bound: 886.6759842
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -886.6765041, upper bound: 886.6765041
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.40
Output dim: 0, lower bound: -886.6765041, upper bound: 886.6765041

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6048745, upper bound: 886.6048745
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6048745, upper bound: 886.6048745
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6728555, upper bound: 886.6727904
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6728555, upper bound: 886.6727873
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6560614, upper bound: 886.6566987
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6592639, upper bound: 886.6564076
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6694328, upper bound: 886.6694451
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6694328, upper bound: 886.6696059
time: 1.21 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.74
Output dim: 0, lower bound: -886.6048745, upper bound: 886.6048745
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.74
Output dim: 0, lower bound: -886.6048745, upper bound: 886.6048745
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -886.6728555, upper bound: 886.6727904
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -886.6728555, upper bound: 886.6727873
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -886.6560614, upper bound: 886.6566987
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -886.6592639, upper bound: 886.6564076
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -886.6694328, upper bound: 886.6694451
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.74
Output dim: 0, lower bound: -886.6694328, upper bound: 886.6696059

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613637, upper bound: 886.6613637
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613637, upper bound: 886.6613637
time: 1.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6538082, upper bound: 886.6532481
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6541112, upper bound: 886.6531940
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6519466, upper bound: 886.6521248
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6519466, upper bound: 886.6521248
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6519902, upper bound: 886.6521248
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6519902, upper bound: 886.6521248
time: 1.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6628989, upper bound: 886.6631984
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6629569, upper bound: 886.6631984
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5623196, upper bound: 886.5623196
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5623196, upper bound: 886.5623262
time: 2.06 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.36 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6613637, upper bound: 886.6613637
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6613637, upper bound: 886.6613637
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6538082, upper bound: 886.6532481
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6541112, upper bound: 886.6531940
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6519466, upper bound: 886.6521248
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6519466, upper bound: 886.6521248
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6519902, upper bound: 886.6521248
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6519902, upper bound: 886.6521248
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6628989, upper bound: 886.6631984
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.6629569, upper bound: 886.6631984
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.5623196, upper bound: 886.5623196
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.36
Output dim: 0, lower bound: -886.5623196, upper bound: 886.5623262

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6566088, upper bound: 886.6566088
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6566088, upper bound: 886.6566088
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613370, upper bound: 886.6613370
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613370, upper bound: 886.6613370
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6532890, upper bound: 886.6531938
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6537210, upper bound: 886.6532375
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6532436, upper bound: 886.6530959
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6540043, upper bound: 886.6530959
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6519466, upper bound: 886.6521248
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6519466, upper bound: 886.6519466
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6364011, upper bound: 886.6364011
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6364011, upper bound: 886.6364011
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6465932
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6465547, upper bound: 886.6464340
time: 1.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6465932
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6464948
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6615658, upper bound: 886.6618629
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6615658, upper bound: 886.6615658
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6629186, upper bound: 886.6628596
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6628596, upper bound: 886.6631601
time: 0.71 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6566088, upper bound: 886.6566088
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6566088, upper bound: 886.6566088
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6613370, upper bound: 886.6613370
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6613370, upper bound: 886.6613370
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6532890, upper bound: 886.6531938
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6537210, upper bound: 886.6532375
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6532436, upper bound: 886.6530959
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6540043, upper bound: 886.6530959
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6519466, upper bound: 886.6521248
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6519466, upper bound: 886.6519466
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6364011, upper bound: 886.6364011
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6364011, upper bound: 886.6364011
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6465932
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6465547, upper bound: 886.6464340
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6465932
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6464948
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6615658, upper bound: 886.6618629
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6615658, upper bound: 886.6615658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6629186, upper bound: 886.6628596
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 0, lower bound: -886.6628596, upper bound: 886.6631601

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6220592, upper bound: 886.6220592
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6220592, upper bound: 886.6220592
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6416537, upper bound: 886.6416537
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6416537, upper bound: 886.6416537
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613370, upper bound: 886.6613370
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6613370, upper bound: 886.6613370
time: 1.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6588926, upper bound: 886.6588926
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6588926, upper bound: 886.6588926
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6464553, upper bound: 886.6463712
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6463712, upper bound: 886.6463760
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6534059, upper bound: 886.6531832
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6531794, upper bound: 886.6532375
time: 1.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6529702, upper bound: 886.6527223
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6530199, upper bound: 886.6527223
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6464272, upper bound: 886.6464272
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6474587, upper bound: 886.6464272
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6397394, upper bound: 886.6397394
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6397394, upper bound: 886.6397394
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6464340
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6464340
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
time: 1.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6417867, upper bound: 886.6417867
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6420326, upper bound: 886.6418246
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6477415, upper bound: 886.6482272
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6477415, upper bound: 886.6477415
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6615658, upper bound: 886.6615658
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6615658, upper bound: 886.6615658
time: 1.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6615278, upper bound: 886.6615278
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6615714, upper bound: 886.6615278
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.4680243, upper bound: 886.4680752
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.4680243, upper bound: 886.4680752
time: 0.95 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6220592, upper bound: 886.6220592
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6220592, upper bound: 886.6220592
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6416537, upper bound: 886.6416537
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6416537, upper bound: 886.6416537
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6613370, upper bound: 886.6613370
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6613370, upper bound: 886.6613370
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6588926, upper bound: 886.6588926
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6588926, upper bound: 886.6588926
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6464553, upper bound: 886.6463712
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6463712, upper bound: 886.6463760
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6534059, upper bound: 886.6531832
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6531794, upper bound: 886.6532375
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6529702, upper bound: 886.6527223
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6530199, upper bound: 886.6527223
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6464272, upper bound: 886.6464272
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6474587, upper bound: 886.6464272
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6397394, upper bound: 886.6397394
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6397394, upper bound: 886.6397394
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6464340
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6464340, upper bound: 886.6464340
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.5883863, upper bound: 886.5883863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6417867, upper bound: 886.6417867
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6420326, upper bound: 886.6418246
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6477415, upper bound: 886.6482272
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6477415, upper bound: 886.6477415
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6615658, upper bound: 886.6615658
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6615658, upper bound: 886.6615658
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6615278, upper bound: 886.6615278
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.6615714, upper bound: 886.6615278
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.4680243, upper bound: 886.4680752
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.59
Output dim: 0, lower bound: -886.4680243, upper bound: 886.4680752

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6285012, upper bound: 886.6285012
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6285012, upper bound: 886.6285012
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6480101, upper bound: 886.6480101
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6480101, upper bound: 886.6480101
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6517130, upper bound: 886.6517130
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6517130, upper bound: 886.6517130
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6553136, upper bound: 886.6553136
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6553136, upper bound: 886.6553136
time: 1.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6463993, upper bound: 886.6463527
time: 1.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6463527, upper bound: 886.6463527
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6369160, upper bound: 886.6369160
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6369160, upper bound: 886.6369160
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6453051, upper bound: 886.6453051
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6453942, upper bound: 886.6453051
time: 1.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6530935, upper bound: 886.6531011
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6536277, upper bound: 886.6531591
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6527016, upper bound: 886.6527016
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6527016, upper bound: 886.6527016
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6527223, upper bound: 886.6527223
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6530121, upper bound: 886.6527223
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6379013, upper bound: 886.6379013
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6379515, upper bound: 886.6379013
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6462953, upper bound: 886.6462953
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6469293, upper bound: 886.6462953
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6333020, upper bound: 886.6333020
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6333020, upper bound: 886.6333020
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6456635, upper bound: 886.6456635
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6456635, upper bound: 886.6456635
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5513092, upper bound: 886.5513092
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.5513092, upper bound: 886.5513092
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6388396, upper bound: 886.6388396
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6388396, upper bound: 886.6388396
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6517513, upper bound: 886.6517513
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6517513, upper bound: 886.6517513
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6606412, upper bound: 886.6606412
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6606412, upper bound: 886.6606412
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6516407, upper bound: 886.6516407
time: 1.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6516407, upper bound: 886.6516406
time: 1.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6572765, upper bound: 886.6572765
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6573103, upper bound: 886.6572765
time: 1.06 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.01 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6285012, upper bound: 886.6285012
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6285012, upper bound: 886.6285012
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6480101, upper bound: 886.6480101
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6480101, upper bound: 886.6480101
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6517130, upper bound: 886.6517130
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6517130, upper bound: 886.6517130
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6553136, upper bound: 886.6553136
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6553136, upper bound: 886.6553136
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6463993, upper bound: 886.6463527
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6463527, upper bound: 886.6463527
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6369160, upper bound: 886.6369160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6369160, upper bound: 886.6369160
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6453051, upper bound: 886.6453051
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6453942, upper bound: 886.6453051
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6530935, upper bound: 886.6531011
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6536277, upper bound: 886.6531591
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6527016, upper bound: 886.6527016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6527016, upper bound: 886.6527016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6527223, upper bound: 886.6527223
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6530121, upper bound: 886.6527223
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6379013, upper bound: 886.6379013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6379515, upper bound: 886.6379013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6462953, upper bound: 886.6462953
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6469293, upper bound: 886.6462953
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6333020, upper bound: 886.6333020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6333020, upper bound: 886.6333020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6456635, upper bound: 886.6456635
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6456635, upper bound: 886.6456635
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.5513092, upper bound: 886.5513092
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.5513092, upper bound: 886.5513092
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6388396, upper bound: 886.6388396
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6388396, upper bound: 886.6388396
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6517513, upper bound: 886.6517513
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6517513, upper bound: 886.6517513
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6606412, upper bound: 886.6606412
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6606412, upper bound: 886.6606412
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6516407, upper bound: 886.6516407
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6516407, upper bound: 886.6516406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6572765, upper bound: 886.6572765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.01
Output dim: 0, lower bound: -886.6573103, upper bound: 886.6572765

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6422150, upper bound: 886.6422150
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6422150, upper bound: 886.6422150
time: 1.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6473553, upper bound: 886.6473553
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6473553, upper bound: 886.6473553
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6507575, upper bound: 886.6507575
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6507575, upper bound: 886.6507575
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6517130, upper bound: 886.6517130
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -886.6517130, upper bound: 886.6517130
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6368430, upper bound: 886.6368430
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -886.6368430, upper bound: 886.6368430
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -390.5541382, 634.1719971, -390.5541382, 634.1719971, -1024.7260742, 1024.7260742
1: -437.5506897, 626.8630371, -437.5506897, 626.8630371, -1064.4134521, 1064.4134521
2: -439.2138672, 620.2371826, -439.2138672, 620.2371826, -1059.4510498, 1059.4510498
3: -536.9467773, 721.8648682, -536.9467773, 721.8648682, -1258.8116455, 1258.8116455
4: -473.3275146, 708.4596558, -473.3275146, 708.4596558, -1181.7871094, 1181.7871094

Time for backsubstitution: 1.89 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=1024.72607421875
rel_dist={0: [-886.6885385920837, 886.6885385920837]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1120.15 seconds
