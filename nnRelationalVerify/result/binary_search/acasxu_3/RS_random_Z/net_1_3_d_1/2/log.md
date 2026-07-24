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
execution time: IAR + LP analysis = 1.45 + 1.16 = 2.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -12.2720720, upper bound: 12.2720720


# Binary Search by BASE starts (time budget: 1197.39 seconds, max iter: 100)

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
Binary search time: 44.81 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1152.58 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.92
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2611369
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2637588
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.56 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1431226, upper bound: 12.1431226
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1444879, upper bound: 12.1431226
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.56 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2611369
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2637588
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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2611369
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2630313
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2611369
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2637588
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.47
Output dim: 0, lower bound: -12.1431226, upper bound: 12.1431226
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.47
Output dim: 0, lower bound: -12.1444879, upper bound: 12.1431226
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2611369
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2637588
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2611369
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.47
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2630313

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2571536
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2587765
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
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2553059, upper bound: 12.2553059
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2553059, upper bound: 12.2553059
time: 0.35 seconds

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
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360853, upper bound: 12.2359740
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
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
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2571536
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2587765
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2553059, upper bound: 12.2553059
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2553059, upper bound: 12.2553059
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2360853, upper bound: 12.2359740
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2307538, upper bound: 12.2306939
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2052992, upper bound: 12.2043959
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2052992, upper bound: 12.2043959
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1251341, upper bound: 12.1357664
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1277348, upper bound: 12.1357664
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
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2307932
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523231
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523511
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2495763
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2495763
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1243573, upper bound: 12.1258222
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1243573, upper bound: 12.1258222
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2284444, upper bound: 12.2280666
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2284444, upper bound: 12.2280666
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
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2048239, upper bound: 12.2043959
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2523676, upper bound: 12.2523231
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2523511, upper bound: 12.2523231
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1243573, upper bound: 12.1258320
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1243573, upper bound: 12.1258320
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2307538, upper bound: 12.2306939
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2052992, upper bound: 12.2043959
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2052992, upper bound: 12.2043959
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.1251341, upper bound: 12.1357664
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.1277348, upper bound: 12.1357664
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2307932
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523231
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523511
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2495763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2495763
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.1243573, upper bound: 12.1258222
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.1243573, upper bound: 12.1258222
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2284444, upper bound: 12.2280666
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2284444, upper bound: 12.2280666
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2048239, upper bound: 12.2043959
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2523676, upper bound: 12.2523231
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.2523511, upper bound: 12.2523231
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.1243573, upper bound: 12.1258320
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.23
Output dim: 0, lower bound: -12.1243573, upper bound: 12.1258320

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985538
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1279426, upper bound: 12.1245982
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1323058, upper bound: 12.1245982
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1122400
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1122400
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1245656, upper bound: 12.1246409
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244348, upper bound: 12.1246409
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1202787
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1229851, upper bound: 12.1202787
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2463123
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2231570, upper bound: 12.2228100
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023312
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1209158
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1209158
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1230319
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1230319
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.41 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985538
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1279426, upper bound: 12.1245982
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1323058, upper bound: 12.1245982
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1122400
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1122400
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1245656, upper bound: 12.1246409
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1244348, upper bound: 12.1246409
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1202787
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1229851, upper bound: 12.1202787
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2463123
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2231570, upper bound: 12.2228100
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023312
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1209158
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1209158
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1230319
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.41
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1230319

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1098423
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1098423
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960883, upper bound: 12.1959978
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1098948
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1098948
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960883, upper bound: 12.1959978
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960883, upper bound: 12.1959978
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1091604, upper bound: 12.1080391
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1091604, upper bound: 12.1080391
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1098423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1098423
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1960883, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1098948
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1098948
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1960883, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1960883, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1091604, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1091604, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005049
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005049
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005049
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005049
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.61
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
time: 0.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
time: 0.35 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2611369
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2637588
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2611369
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2630313
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2576816, upper bound: 12.2567650
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2563716, upper bound: 12.2587765
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.58 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2587765, upper bound: 12.2563687
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2571536, upper bound: 12.2577425
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2611369
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2637588
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2611369
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2630313
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -12.2576816, upper bound: 12.2567650
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -12.2563716, upper bound: 12.2587765
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -12.2587765, upper bound: 12.2563687
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 0, lower bound: -12.2571536, upper bound: 12.2577425

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1429248, upper bound: 12.1338659
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1429248, upper bound: 12.1338659
time: 0.36 seconds

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
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360853, upper bound: 12.2359740
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
time: 0.37 seconds

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
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
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
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2576816, upper bound: 12.2555657
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2567650
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2563716, upper bound: 12.2555657
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2587765
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
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2516205, upper bound: 12.2504363
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2516205, upper bound: 12.2504363
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1429248, upper bound: 12.1338659
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1429248, upper bound: 12.1338659
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2360853, upper bound: 12.2359740
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2576816, upper bound: 12.2555657
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2567650
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2563716, upper bound: 12.2555657
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2587765
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2516205, upper bound: 12.2504363
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2516205, upper bound: 12.2504363
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1256179, upper bound: 12.1258154
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1259664, upper bound: 12.1258154
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2048239
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.33 seconds

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
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1428455, upper bound: 12.1319246
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1442654, upper bound: 12.1319246
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2053896
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1371507, upper bound: 12.1245892
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1251341, upper bound: 12.1245892
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1356307, upper bound: 12.1277373
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1356307, upper bound: 12.1277373
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2516205
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2516205
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2516205, upper bound: 12.2495763
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2495763
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2464735
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2463123, upper bound: 12.2462027
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462621, upper bound: 12.2462027
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.39 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.1256179, upper bound: 12.1258154
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.1259664, upper bound: 12.1258154
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2048239
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.1428455, upper bound: 12.1319246
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.1442654, upper bound: 12.1319246
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2053896
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.1371507, upper bound: 12.1245892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.1251341, upper bound: 12.1245892
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.1356307, upper bound: 12.1277373
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.1356307, upper bound: 12.1277373
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2516205
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2516205
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2516205, upper bound: 12.2495763
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2495763, upper bound: 12.2495763
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2464735
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2463123, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2462621, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1118837
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1118837
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1147239, upper bound: 12.1163992
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1147239, upper bound: 12.1163992
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2231570
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1379961
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1379961
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2464735
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1121954, upper bound: 12.1144203
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1121954, upper bound: 12.1144203
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960221, upper bound: 12.1960118
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462621, upper bound: 12.2462027
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1148806, upper bound: 12.1183628
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1148806, upper bound: 12.1183628
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.02 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1118837
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1118837
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1147239, upper bound: 12.1163992
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1147239, upper bound: 12.1163992
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2231570
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1379961
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1379961
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2464735
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1121954, upper bound: 12.1144203
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1121954, upper bound: 12.1144203
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1960221, upper bound: 12.1960118
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462621, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1148806, upper bound: 12.1183628
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.1148806, upper bound: 12.1183628
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.02
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1960883
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1960883
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1225258
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1225258
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1216515, upper bound: 12.1240774
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1216515, upper bound: 12.1240774
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1217393, upper bound: 12.1186856
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1217393, upper bound: 12.1186856
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1216130, upper bound: 12.1367774
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1367774
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1116116, upper bound: 12.1080327
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1080327
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1116116, upper bound: 12.1080327
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1080327
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1115050, upper bound: 12.1080327
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1131008, upper bound: 12.1080327
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1111252, upper bound: 12.1131898
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1118348, upper bound: 12.1131898
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960068, upper bound: 12.1959978
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1143132
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1143132
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1133111
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1133111
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1115050, upper bound: 12.1133111
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1115050, upper bound: 12.1133111
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1960883
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1960883
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1225258
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1225258
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1216515, upper bound: 12.1240774
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1216515, upper bound: 12.1240774
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1217393, upper bound: 12.1186856
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1217393, upper bound: 12.1186856
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1216130, upper bound: 12.1367774
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1182772, upper bound: 12.1367774
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1116116, upper bound: 12.1080327
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1080327
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1116116, upper bound: 12.1080327
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1080327
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1115050, upper bound: 12.1080327
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1131008, upper bound: 12.1080327
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1111252, upper bound: 12.1131898
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1118348, upper bound: 12.1131898
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1960068, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1143132
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1143132
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1133111
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1133111
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1115050, upper bound: 12.1133111
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1115050, upper bound: 12.1133111
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005494, upper bound: 12.1000990
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005494, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005049, upper bound: 12.1000990
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005592, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.13 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1005494, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1005494, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1005049, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1005592, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.13
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1367013, upper bound: 12.1367013
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1367013, upper bound: 12.1367013
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 0.68
Output dim: 0, lower bound: -12.1367013, upper bound: 12.1367013
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 0.68
Output dim: 0, lower bound: -12.1367013, upper bound: 12.1367013
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
time: 0.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.78 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
time: 0.34 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
time: 0.37 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.89 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.89
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.89
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.89
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.89
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1431226, upper bound: 12.1431226
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1444879, upper bound: 12.1431226
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2576816, upper bound: 12.2567650
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2563716, upper bound: 12.2587765
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360853, upper bound: 12.2359740
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359776
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 0, lower bound: -12.1431226, upper bound: 12.1431226
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.75
Output dim: 0, lower bound: -12.1444879, upper bound: 12.1431226
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -12.2576816, upper bound: 12.2567650
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -12.2563716, upper bound: 12.2587765
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -12.2360853, upper bound: 12.2359740
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359776

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2048239
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2052992, upper bound: 12.2043959
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281133, upper bound: 12.1285377
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1281133, upper bound: 12.1285377
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1373577, upper bound: 12.1287992
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1373577, upper bound: 12.1287992
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2504363, upper bound: 12.2516205
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2504363, upper bound: 12.2516205
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
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360853, upper bound: 12.2359740
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2048239, upper bound: 12.2043959
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2048239
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2052992, upper bound: 12.2043959
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1281133, upper bound: 12.1285377
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1281133, upper bound: 12.1285377
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1373577, upper bound: 12.1287992
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1373577, upper bound: 12.1287992
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2504363, upper bound: 12.2516205
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2504363, upper bound: 12.2516205
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2360853, upper bound: 12.2359740
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2359740, upper bound: 12.2359740
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2048239, upper bound: 12.2043959

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1246141, upper bound: 12.1238446
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1246141, upper bound: 12.1238446
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023312, upper bound: 12.2023291
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2229884, upper bound: 12.2229884
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2229884, upper bound: 12.2231570
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2229884, upper bound: 12.2229884
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2229884, upper bound: 12.2231570
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1443811, upper bound: 12.1319967
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1443811, upper bound: 12.1319967
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2052992
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2052992
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1989767, upper bound: 12.1985486
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1238446, upper bound: 12.1246141
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1238446, upper bound: 12.1246141
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 1.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.1246141, upper bound: 12.1238446
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.1246141, upper bound: 12.1238446
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.2023312, upper bound: 12.2023291
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.2229884, upper bound: 12.2229884
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.2229884, upper bound: 12.2231570
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.2229884, upper bound: 12.2229884
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.2229884, upper bound: 12.2231570
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.1443811, upper bound: 12.1319967
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.1443811, upper bound: 12.1319967
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2052992
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2052992
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.1989767, upper bound: 12.1985486
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.1238446, upper bound: 12.1246141
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 1.93
Output dim: 0, lower bound: -12.1238446, upper bound: 12.1246141

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1273245, upper bound: 12.1371338
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1292194, upper bound: 12.1371338
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1331633, upper bound: 12.1252059
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1273245, upper bound: 12.1252059
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2231570
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1994659
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1993344
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1989767, upper bound: 12.1985486
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985538, upper bound: 12.1985486
time: 0.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 1.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1960118, upper bound: 12.1960118
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1273245, upper bound: 12.1371338
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1292194, upper bound: 12.1371338
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1331633, upper bound: 12.1252059
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1273245, upper bound: 12.1252059
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2231570
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1994659
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1993344
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1989767, upper bound: 12.1985486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -12.1985538, upper bound: 12.1985486

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1131898, upper bound: 12.1118348
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1131898, upper bound: 12.1118348
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1098423
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1098423
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1340323, upper bound: 12.1182772
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1340323, upper bound: 12.1182772
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1217393, upper bound: 12.1186856
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1217393, upper bound: 12.1186856
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1272333, upper bound: 12.1252829
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1272333, upper bound: 12.1252829
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960532, upper bound: 12.1959978
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054067
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054067
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1131898, upper bound: 12.1118348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1131898, upper bound: 12.1118348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1098423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1098423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1340323, upper bound: 12.1182772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1340323, upper bound: 12.1182772
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1217393, upper bound: 12.1186856
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1217393, upper bound: 12.1186856
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1272333, upper bound: 12.1252829
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1272333, upper bound: 12.1252829
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1960532, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054067
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054067
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.10
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005592, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005592, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005592, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1005592, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1005592, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1005592, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2000000, mid=0.1875000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 4) starts
Candidate diff: 0.1937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2635057, upper bound: 12.2674811
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2635057, upper bound: 12.2635057
time: 0.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -12.2635057, upper bound: 12.2674811
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -12.2635057, upper bound: 12.2635057

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
time: 0.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.90 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.90
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.90
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.90
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.90
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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.56 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2097017, upper bound: 12.2100296
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2069307, upper bound: 12.2124949
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.57 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140079, upper bound: 12.2080070
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2080070, upper bound: 12.2122007
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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.56 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2164699, upper bound: 12.2135927
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2184025, upper bound: 12.2126676
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2119687
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -12.2097017, upper bound: 12.2100296
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -12.2069307, upper bound: 12.2124949
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -12.2140079, upper bound: 12.2080070
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -12.2080070, upper bound: 12.2122007
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -12.2164699, upper bound: 12.2135927
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -12.2184025, upper bound: 12.2126676

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056540
time: 0.34 seconds

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
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2051237, upper bound: 12.2038144
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056328
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056540
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2041058, upper bound: 12.2025568
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2025568, upper bound: 12.2025568
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1317324, upper bound: 12.1418470
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1317324, upper bound: 12.1418470
time: 0.38 seconds

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
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2100192, upper bound: 12.2100192
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2100192, upper bound: 12.2100351
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
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2071711, upper bound: 12.2071711
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2071711, upper bound: 12.2071711
time: 0.36 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2108541, upper bound: 12.2108541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2108905, upper bound: 12.2119391
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056540
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2051237, upper bound: 12.2038144
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056328
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056540
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2041058, upper bound: 12.2025568
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2025568, upper bound: 12.2025568
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1317324, upper bound: 12.1418470
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1317324, upper bound: 12.1418470
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2100192, upper bound: 12.2100192
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2100192, upper bound: 12.2100351
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2071711, upper bound: 12.2071711
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2071711, upper bound: 12.2071711

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2034859, upper bound: 12.2025568
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2025568, upper bound: 12.2028639
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
time: 0.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2046527
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
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
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2051237, upper bound: 12.2038144
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2034202, upper bound: 12.2027457
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2046137
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056540
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056445
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959671
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2041058, upper bound: 12.2025568
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2028639, upper bound: 12.2025568
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1300037, upper bound: 12.1323752
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1300037, upper bound: 12.1323752
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2100192, upper bound: 12.2100351
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2100192, upper bound: 12.2100192
time: 0.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1401425, upper bound: 12.1266458
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1401425, upper bound: 12.1266458
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
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2061522, upper bound: 12.2061522
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2061522, upper bound: 12.2061522
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2034859, upper bound: 12.2025568
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2025568, upper bound: 12.2028639
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.1322509, upper bound: 12.1325999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2046527
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2051237, upper bound: 12.2038144
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2034202, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2046137
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056540
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056445
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959671
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2041058, upper bound: 12.2025568
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2028639, upper bound: 12.2025568
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.1300037, upper bound: 12.1323752
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.1300037, upper bound: 12.1323752
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2100192, upper bound: 12.2100351
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2100192, upper bound: 12.2100192
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.1401425, upper bound: 12.1266458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.1401425, upper bound: 12.1266458
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2061522, upper bound: 12.2061522
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -12.2061522, upper bound: 12.2061522

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1318828, upper bound: 12.1246927
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1318828, upper bound: 12.1246927
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1963999
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165710, upper bound: 12.1155861
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165710, upper bound: 12.1155861
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1967570
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1248743, upper bound: 12.1167045
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1248743, upper bound: 12.1167045
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1213369, upper bound: 12.1169901
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1213369, upper bound: 12.1169901
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2034202, upper bound: 12.2027457
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2046137
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1980492
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1156412, upper bound: 12.1246530
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1156303, upper bound: 12.1246530
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1980845, upper bound: 12.1958826
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949669
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1317653, upper bound: 12.1296425
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1317653, upper bound: 12.1296425
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2017751, upper bound: 12.2017751
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2019951, upper bound: 12.2017751
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1300037, upper bound: 12.1313961
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1300037, upper bound: 12.1313961
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2031003
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1253958, upper bound: 12.1282102
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1253958, upper bound: 12.1282102
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1997935, upper bound: 12.1997935
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1997935, upper bound: 12.1997935
time: 0.39 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1318828, upper bound: 12.1246927
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1318828, upper bound: 12.1246927
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1963999
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1165710, upper bound: 12.1155861
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1165710, upper bound: 12.1155861
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1967570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1248743, upper bound: 12.1167045
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1248743, upper bound: 12.1167045
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1213369, upper bound: 12.1169901
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1213369, upper bound: 12.1169901
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2034202, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2046137
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1980492
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1156412, upper bound: 12.1246530
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1156303, upper bound: 12.1246530
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1980845, upper bound: 12.1958826
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949669
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1317653, upper bound: 12.1296425
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1317653, upper bound: 12.1296425
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2017751, upper bound: 12.2017751
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2019951, upper bound: 12.2017751
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1300037, upper bound: 12.1313961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1300037, upper bound: 12.1313961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2031003
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1253958, upper bound: 12.1282102
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1253958, upper bound: 12.1282102
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1997935, upper bound: 12.1997935
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.07
Output dim: 0, lower bound: -12.1997935, upper bound: 12.1997935

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1953060
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244905, upper bound: 12.1155263
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244905, upper bound: 12.1155263
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1234569, upper bound: 12.1155534
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1234569, upper bound: 12.1155534
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1190247
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1190247
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1967567
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1967570
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1211409, upper bound: 12.1169901
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1211409, upper bound: 12.1169901
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1229026, upper bound: 12.1138428
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1229026, upper bound: 12.1138428
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1962939, upper bound: 12.1949473
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949669, upper bound: 12.1949473
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1953060
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1965270
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1967567
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244024, upper bound: 12.1155263
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244024, upper bound: 12.1155263
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949669
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1237081, upper bound: 12.1295368
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1240737, upper bound: 12.1295368
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1953060, upper bound: 12.1949473
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1200975, upper bound: 12.1161852
time: 0.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1200975, upper bound: 12.1161852
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1961852
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1184075, upper bound: 12.1222135
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1184075, upper bound: 12.1222135
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165806, upper bound: 12.1238743
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165806, upper bound: 12.1238743
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1953060
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1244905, upper bound: 12.1155263
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1244905, upper bound: 12.1155263
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1234569, upper bound: 12.1155534
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1234569, upper bound: 12.1155534
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1190247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1190247
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1967567
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1967570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1211409, upper bound: 12.1169901
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1211409, upper bound: 12.1169901
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1229026, upper bound: 12.1138428
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1229026, upper bound: 12.1138428
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1962939, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949669, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1953060
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1965270
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1967567
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949840
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1244024, upper bound: 12.1155263
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1244024, upper bound: 12.1155263
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949669
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1237081, upper bound: 12.1295368
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1240737, upper bound: 12.1295368
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1953060, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1200975, upper bound: 12.1161852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1200975, upper bound: 12.1161852
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1961852
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1184075, upper bound: 12.1222135
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1184075, upper bound: 12.1222135
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1165806, upper bound: 12.1238743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1165806, upper bound: 12.1238743

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1133860
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1222663, upper bound: 12.1133860
time: 0.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1171492
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1171492
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1199299
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1199299
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226314
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226314
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1232912
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1233138
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1151336, upper bound: 12.1172306
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1151336, upper bound: 12.1172306
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1209424
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1209424
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1211607
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1211607
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1235521, upper bound: 12.1133860
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1235521, upper bound: 12.1133860
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1142560
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1222663, upper bound: 12.1142560
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1134402, upper bound: 12.1133860
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1134402, upper bound: 12.1133860
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1134823
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1134823
time: 0.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1184343
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1184343
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1193132
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1193132
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1234653
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1234653
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1199299
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1199299
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1209628
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1209628
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1195195
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1195195
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1163978
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1163978
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226314
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226314
time: 0.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1208344
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1208344
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1208344, upper bound: 12.1133860
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182315, upper bound: 12.1133860
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1226314, upper bound: 12.1133860
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1226314, upper bound: 12.1133860
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1222663
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1222663
time: 0.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1168856, upper bound: 12.1140717
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1168856, upper bound: 12.1140717
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1159600, upper bound: 12.1173415
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1173415
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1184343, upper bound: 12.1133860
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1145455, upper bound: 12.1176610
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1176610
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1222663
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1222663
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1235521
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1235521
time: 0.62 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1133860
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1222663, upper bound: 12.1133860
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1171492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1171492
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1199299
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1199299
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1232912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1233138
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1151336, upper bound: 12.1172306
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1151336, upper bound: 12.1172306
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1209424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1160068, upper bound: 12.1209424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1211607
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1211607
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1235521, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1235521, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1142560
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1222663, upper bound: 12.1142560
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1134402, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1134402, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1134823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1134823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1184343
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1184343
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1193132
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1193132
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1234653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1234653
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1199299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1199299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1209628
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1209628
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1195195
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1195195
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1163978
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1163978
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226314
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226314
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1208344
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1208344
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1208344, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1182315, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1226314, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1226314, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1222663
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1222663
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1168856, upper bound: 12.1140717
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1168856, upper bound: 12.1140717
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1159600, upper bound: 12.1173415
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1173415
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1184343, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1145455, upper bound: 12.1176610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1176610
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1222663
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1222663
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1235521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1235521
Binary search (step 4): status=Status.VERIFIED, low=0.1937500, high=0.2000000, mid=0.1937500, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 5) starts
Candidate diff: 0.1968750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2635057, upper bound: 12.2674811
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2674811, upper bound: 12.2635057
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -12.2635057, upper bound: 12.2674811
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -12.2674811, upper bound: 12.2635057

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.91 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.91
Output dim: 0, lower bound: -12.2145008, upper bound: 12.2192610
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.91
Output dim: 0, lower bound: -12.2159045, upper bound: 12.2192610
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.91
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.91
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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2069307, upper bound: 12.2081291
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2069307, upper bound: 12.2124949
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2119687
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2140079, upper bound: 12.2080070
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2080070, upper bound: 12.2122007
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2124949, upper bound: 12.2069307
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2081291, upper bound: 12.2079887
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -12.2069307, upper bound: 12.2081291
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -12.2069307, upper bound: 12.2124949
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -12.2110927, upper bound: 12.2119391
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -12.2110067, upper bound: 12.2119687
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -12.2140079, upper bound: 12.2080070
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -12.2080070, upper bound: 12.2122007
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -12.2124949, upper bound: 12.2069307
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -12.2081291, upper bound: 12.2079887

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1353178, upper bound: 12.1252730
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1353178, upper bound: 12.1252730
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2018287, upper bound: 12.2013598
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2013598, upper bound: 12.2080098
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2051237, upper bound: 12.2038144
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056328
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1330838
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1330838
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
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2080098, upper bound: 12.2013598
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2055183, upper bound: 12.2013598
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2072981, upper bound: 12.2083143
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2072981, upper bound: 12.2122007
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2056328, upper bound: 12.2038144
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1252730, upper bound: 12.1353178
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1252730, upper bound: 12.1353178
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1353178, upper bound: 12.1252730
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1353178, upper bound: 12.1252730
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2018287, upper bound: 12.2013598
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2013598, upper bound: 12.2080098
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2051237, upper bound: 12.2038144
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2056328
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1330838
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1332431, upper bound: 12.1330838
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2080098, upper bound: 12.2013598
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2055183, upper bound: 12.2013598
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2072981, upper bound: 12.2083143
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2072981, upper bound: 12.2122007
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2038144, upper bound: 12.2038144
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.2056328, upper bound: 12.2038144
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1252730, upper bound: 12.1353178
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.92
Output dim: 0, lower bound: -12.1252730, upper bound: 12.1353178

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1247751, upper bound: 12.1215081
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1277178, upper bound: 12.1215081
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2033345
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2080098
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1248743, upper bound: 12.1190210
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1248743, upper bound: 12.1190210
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2046137
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1344995, upper bound: 12.1220390
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1344995, upper bound: 12.1220390
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959671
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959702
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2009592
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2027774
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2018415
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2061947
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244708, upper bound: 12.1182961
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244708, upper bound: 12.1182961
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244708, upper bound: 12.1182961
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1244708, upper bound: 12.1182961
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.02 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1247751, upper bound: 12.1215081
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1277178, upper bound: 12.1215081
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2033345
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2080098
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1248743, upper bound: 12.1190210
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1248743, upper bound: 12.1190210
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2046137
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.2027457, upper bound: 12.2027457
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1344995, upper bound: 12.1220390
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1344995, upper bound: 12.1220390
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959671
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959702
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2009592
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2027774
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2018415
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.2009592, upper bound: 12.2061947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1244708, upper bound: 12.1182961
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1244708, upper bound: 12.1182961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1244708, upper bound: 12.1182961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.02
Output dim: 0, lower bound: -12.1244708, upper bound: 12.1182961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1235329, upper bound: 12.1253072
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1230742, upper bound: 12.1253072
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182268, upper bound: 12.1339216
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182268, upper bound: 12.1339216
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1135492, upper bound: 12.1237464
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1135492, upper bound: 12.1237464
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1135492, upper bound: 12.1211670
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1135492, upper bound: 12.1211670
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949669
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959623
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959702
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1978940
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1189025, upper bound: 12.1247751
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1220491, upper bound: 12.1247751
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1997935, upper bound: 12.2050587
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1997935, upper bound: 12.2023080
time: 0.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1235329, upper bound: 12.1253072
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1230742, upper bound: 12.1253072
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1182268, upper bound: 12.1339216
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1182268, upper bound: 12.1339216
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1135492, upper bound: 12.1237464
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1135492, upper bound: 12.1237464
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1135492, upper bound: 12.1211670
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1135492, upper bound: 12.1211670
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949669
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959623
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1959702
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1958826
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1958826, upper bound: 12.1978940
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1189025, upper bound: 12.1247751
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1220491, upper bound: 12.1247751
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1997935, upper bound: 12.2050587
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -12.1997935, upper bound: 12.2023080

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949669
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1168856, upper bound: 12.1173415
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1168856, upper bound: 12.1173415
time: 0.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1155263, upper bound: 12.1225305
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1179637, upper bound: 12.1225305
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949834
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182972, upper bound: 12.1191545
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1182972, upper bound: 12.1191545
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1966546
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165806, upper bound: 12.1339282
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165806, upper bound: 12.1339282
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1174465, upper bound: 12.1290357
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1174465, upper bound: 12.1290357
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.01 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949669
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1168856, upper bound: 12.1173415
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1168856, upper bound: 12.1173415
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1155263, upper bound: 12.1225305
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1179637, upper bound: 12.1225305
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949834
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1182972, upper bound: 12.1191545
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1182972, upper bound: 12.1191545
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1966546
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1949473, upper bound: 12.1949473
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1165806, upper bound: 12.1339282
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1165806, upper bound: 12.1339282
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1174465, upper bound: 12.1290357
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.01
Output dim: 0, lower bound: -12.1174465, upper bound: 12.1290357

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1222663
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1222663
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1222663
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1222663
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1176610
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1145455, upper bound: 12.1176610
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1194489, upper bound: 12.1133860
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1194489, upper bound: 12.1133860
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1193132, upper bound: 12.1133860
time: 0.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165888, upper bound: 12.1133860
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226216
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226216
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165888, upper bound: 12.1208874
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1165888, upper bound: 12.1208874
time: 0.41 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.51 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1222663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1222663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1222663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1222663
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1142560, upper bound: 12.1176610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1145455, upper bound: 12.1176610
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1194489, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1194489, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1193132, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1165888, upper bound: 12.1133860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226216
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1133860, upper bound: 12.1226216
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1165888, upper bound: 12.1208874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.51
Output dim: 0, lower bound: -12.1165888, upper bound: 12.1208874
Binary search (step 5): status=Status.VERIFIED, low=0.1968750, high=0.2000000, mid=0.1968750, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 6) starts
Candidate diff: 0.1984375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
time: 0.38 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -12.2683498, upper bound: 12.2683498

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
time: 0.36 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
time: 0.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.41 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2637588
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -12.2637588, upper bound: 12.2630313

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2577425, upper bound: 12.2571536
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2563687, upper bound: 12.2587765
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.55 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2611369
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2637588
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.54 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2587765, upper bound: 12.2563687
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2571536, upper bound: 12.2577425
time: 0.36 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2577425, upper bound: 12.2571536
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2563687, upper bound: 12.2587765
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2630313, upper bound: 12.2611369
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2611369, upper bound: 12.2637588
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2587765, upper bound: 12.2563687
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2571536, upper bound: 12.2577425

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1382773, upper bound: 12.1363871
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1382773, upper bound: 12.1363871
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1326077, upper bound: 12.1395073
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1326077, upper bound: 12.1395073
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
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2053896
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.34 seconds

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
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1359699, upper bound: 12.1326077
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1373577, upper bound: 12.1326077
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2571536, upper bound: 12.2555657
time: 0.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2577425
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 1.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.1382773, upper bound: 12.1363871
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.1382773, upper bound: 12.1363871
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.1326077, upper bound: 12.1395073
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.1326077, upper bound: 12.1395073
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2053896
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2565693, upper bound: 12.2565693
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.1359699, upper bound: 12.1326077
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.1373577, upper bound: 12.1326077
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2571536, upper bound: 12.2555657
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 1.98
Output dim: 0, lower bound: -12.2555657, upper bound: 12.2577425

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523231
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523231
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1251518, upper bound: 12.1239109
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1235679, upper bound: 12.1239109
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
time: 0.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523511
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523676
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1256179, upper bound: 12.1258222
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1256179, upper bound: 12.1258222
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2307538
time: 0.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523231
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523231
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.1251518, upper bound: 12.1239109
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.1235679, upper bound: 12.1239109
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2043959, upper bound: 12.2043959
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523511
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2523231, upper bound: 12.2523676
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.1256179, upper bound: 12.1258222
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.1256179, upper bound: 12.1258222
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2505655, upper bound: 12.2505655
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2306939
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.32
Output dim: 0, lower bound: -12.2306939, upper bound: 12.2307538

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1230698, upper bound: 12.1193878
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1230698, upper bound: 12.1193878
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1180307, upper bound: 12.1158473
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1147239, upper bound: 12.1158473
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985955
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462621
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1234707
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1234707
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1137142, upper bound: 12.1168246
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1137142, upper bound: 12.1168246
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1988878
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
time: 0.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1245980, upper bound: 12.1380530
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1245980, upper bound: 12.1380530
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1230698, upper bound: 12.1193878
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1230698, upper bound: 12.1193878
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2023291, upper bound: 12.2023291
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1180307, upper bound: 12.1158473
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1147239, upper bound: 12.1158473
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985955
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1110783, upper bound: 12.1110783
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462621
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1234707
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1193878, upper bound: 12.1234707
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2462027, upper bound: 12.2462027
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1137142, upper bound: 12.1168246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1137142, upper bound: 12.1168246
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1988878
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1985486, upper bound: 12.1985486
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.2228100, upper bound: 12.2228100
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1245980, upper bound: 12.1380530
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.03
Output dim: 0, lower bound: -12.1245980, upper bound: 12.1380530

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1113252, upper bound: 12.1080327
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1080327
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1119797
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1119797
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1960883, upper bound: 12.1959978
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1106543, upper bound: 12.1119797
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1119797
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1100406
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1100406
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054067
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054067
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054036
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054036
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080391, upper bound: 12.1080391
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1052825
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1113252, upper bound: 12.1080327
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1080327
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1119797
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1119797
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1960883, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1106543, upper bound: 12.1119797
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1119797
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1100406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1080327, upper bound: 12.1100406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054067
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054067
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054036
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1052825, upper bound: 12.1054036
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.05
Output dim: 0, lower bound: -12.1959978, upper bound: 12.1959978

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 4
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 4

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005974, upper bound: 12.1000990
time: 0.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1005974, upper bound: 12.1000990
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311
1: -5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291
2: -7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632
3: -2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622
4: -10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 45
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 38

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1

### Candidate
type: RSZ, layer: 3, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
time: 0.38 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.11 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1005494
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1005974, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1005974, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.11
Output dim: 0, lower bound: -12.1000990, upper bound: 12.1000990
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
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

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

Time for backsubstitution: 1.19 seconds
Binary search (step 7): status=Status.UNKNOWN, low=0.1984375, high=0.1992187, mid=0.1992187, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1984374881722033
execution time: 1152.64 seconds
