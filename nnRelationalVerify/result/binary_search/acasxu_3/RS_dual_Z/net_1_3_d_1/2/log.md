## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 12.14935128


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311)
1: (-5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291)
2: (-7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632)
3: (-2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622)
4: (-10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130)

## BASE Result
execution time: IAR + LP analysis = 1.30 + 1.14 = 2.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -12.2720720, upper bound: 12.2720720


# Binary Search by BASE starts (time budget: 1197.56 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=13.458332061767578
rel_dist={0: [-12.27133016030539, 12.27133016030539]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=13.458332061767578
rel_dist={0: [-12.270288718386343, 12.270288718386343]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=13.458332061767578
rel_dist={0: [-12.269294519006817, 12.269294519006817]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=13.458332061767578
rel_dist={0: [-12.267428912659181, 12.267428912668205]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0031250, mid=0.0031250, abs_max=13.458332061767578
rel_dist={0: [-12.266075169862887, 12.266075169858812]}

## Binary search (step 6) starts
Candidate diff: 0.0015625


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0015625, mid=0.0015625, abs_max=13.458332061767578
rel_dist={0: [-12.265364319813424, 12.265364319813422]}

## Binary search (step 7) starts
Candidate diff: 0.0007812


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007812, mid=0.0007812, abs_max=13.458332061767578
rel_dist={0: [-12.264960605505483, 12.264960605501642]}

## Binary search (step 8) starts
Candidate diff: 0.0003906


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003906, mid=0.0003906, abs_max=13.458332061767578
rel_dist={0: [-12.264476255980234, 12.264476255980234]}

## Binary search (step 9) starts
Candidate diff: 0.0001953


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001953, mid=0.0001953, abs_max=13.458332061767578
rel_dist={0: [-12.264223046454173, 12.264223046453044]}

## Binary search (step 10) starts
Candidate diff: 0.0000977


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000977, mid=0.0000977, abs_max=13.458332061767578
rel_dist={0: [-12.264089660028645, 12.264089660028077]}

## Binary search (step 11) starts
Candidate diff: 0.0000488


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000488, mid=0.0000488, abs_max=13.458332061767578
rel_dist={0: [-12.264022967421505, 12.264022967420932]}

## Binary search (step 12) starts
Candidate diff: 0.0000244


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000244, mid=0.0000244, abs_max=13.458332061767578
rel_dist={0: [-12.263989620702842, 12.263989620702837]}

## Binary search (step 13) starts
Candidate diff: 0.0000122


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000122, mid=0.0000122, abs_max=13.458332061767578
rel_dist={0: [-12.263972948045687, 12.263972948045613]}

## Binary search (step 14) starts
Candidate diff: 0.0000061


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000061, mid=0.0000061, abs_max=13.458332061767578
rel_dist={0: [-12.263964611984576, 12.263964611984576]}

## Binary search (step 15) starts
Candidate diff: 0.0000031


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000031, mid=0.0000031, abs_max=13.458332061767578
rel_dist={0: [-12.263960445676606, 12.263960445676588]}

## Binary search (step 16) starts
Candidate diff: 0.0000015


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000015, mid=0.0000015, abs_max=13.458332061767578
rel_dist={0: [-12.26395932349288, 12.263958364872]}

## Binary search (step 17) starts
Candidate diff: 0.0000008


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000008, mid=0.0000008, abs_max=13.458332061767578
rel_dist={0: [-12.263957452500073, 12.26395781929547]}

## Binary Search Result
Binary search time: 43.78 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1153.78 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2159045
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.69
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1319848
time: 0.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1319848
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1281971
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.82 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1297288
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.93 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.93
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.93
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.96 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.96
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.96
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.34 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.15
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2000000, mid=0.1875000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 4) starts
Candidate diff: 0.1937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2132805
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.83 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.91 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.21
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 4): status=Status.VERIFIED, low=0.1937500, high=0.2000000, mid=0.1937500, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 5) starts
Candidate diff: 0.1968750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.89 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1297288
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1297288
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1321551
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.24
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 5): status=Status.VERIFIED, low=0.1968750, high=0.2000000, mid=0.1968750, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 6) starts
Candidate diff: 0.1984375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1319848
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1281971
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1322509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.44 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 6): status=Status.VERIFIED, low=0.1984375, high=0.2000000, mid=0.1984375, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 7) starts
Candidate diff: 0.1992187


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.92 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.33
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.79 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 7): status=Status.VERIFIED, low=0.1992187, high=0.2000000, mid=0.1992187, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 8) starts
Candidate diff: 0.1996094


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.04
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1330838
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.41 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 8): status=Status.VERIFIED, low=0.1996094, high=0.2000000, mid=0.1996094, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 9) starts
Candidate diff: 0.1998047


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.80 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.77
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1330838
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.27
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 9): status=Status.VERIFIED, low=0.1998047, high=0.2000000, mid=0.1998047, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 10) starts
Candidate diff: 0.1999023


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.99
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.99
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.79 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.82 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.81 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.09
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 10): status=Status.VERIFIED, low=0.1999023, high=0.2000000, mid=0.1999023, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 11) starts
Candidate diff: 0.1999512


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.38 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.08
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
Binary search (step 11): status=Status.VERIFIED, low=0.1999512, high=0.2000000, mid=0.1999512, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 12) starts
Candidate diff: 0.1999756


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2145008
time: 0.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119391
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1319848
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.20
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 12): status=Status.VERIFIED, low=0.1999756, high=0.2000000, mid=0.1999756, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 13) starts
Candidate diff: 0.1999878


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.40 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1322509
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
time: 0.36 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
Binary search (step 13): status=Status.VERIFIED, low=0.1999878, high=0.2000000, mid=0.1999878, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 14) starts
Candidate diff: 0.1999939


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2145008
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.37
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1330838
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.19
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 14): status=Status.VERIFIED, low=0.1999939, high=0.2000000, mid=0.1999939, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 15) starts
Candidate diff: 0.1999969


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.76 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1319848
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1319848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1264024, upper bound: 12.1325999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1281971
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1330838, upper bound: 12.1322509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1321551, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1297288
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1321551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1281971, upper bound: 12.1330838
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1264024
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1325999, upper bound: 12.1322509
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1319848, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.22
Output dim: 0, lower bound: -12.1297288, upper bound: 12.1332431
Binary search (step 15): status=Status.VERIFIED, low=0.1999969, high=0.2000000, mid=0.1999969, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 16) starts
Candidate diff: 0.1999985


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.98 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.98
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.98
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2145008
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.79 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.92 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.86 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
time: 0.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2140242
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2159045
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2140242
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -12.2132805, upper bound: 12.2192610
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2132805
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2118872
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2110927
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2118872, upper bound: 12.2110067
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2110553
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2110553, upper bound: 12.2108541
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2118872
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2119687, upper bound: 12.2108541
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.20
Output dim: 0, lower bound: -12.2119391, upper bound: 12.2108905
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.20
Output dim: 0, lower bound: -12.2140242, upper bound: 12.2145008
Binary search (step 16): status=Status.UNKNOWN, low=0.1999969, high=0.1999985, mid=0.1999985, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.19999693632144044
execution time: 1154.12 seconds
