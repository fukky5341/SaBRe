## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 95.41947638718


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396)
1: (-25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411)
2: (-22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495)
3: (-42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475)
4: (-33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989)

## BASE Result
execution time: IAR + LP analysis = 1.75 + 1.61 = 3.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -95.4385641, upper bound: 95.4385641


# Binary Search by BASE starts (time budget: 1196.64 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=107.93389892578125
rel_dist={4: [-95.43841502229098, 95.438415022291]}

## Binary search (step 3) starts
Candidate diff: 0.0208333


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0208333, mid=0.0208333, abs_max=107.93389892578125
rel_dist={4: [-95.43801776385524, 95.43801776385524]}

## Binary search (step 4) starts
Candidate diff: 0.0104167


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0104167, mid=0.0104167, abs_max=107.93389892578125
rel_dist={4: [-95.43743645603627, 95.43743645603627]}

## Binary search (step 5) starts
Candidate diff: 0.0052083


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0052083, mid=0.0052083, abs_max=107.93389892578125
rel_dist={4: [-95.43685766005481, 95.43685766005478]}

## Binary search (step 6) starts
Candidate diff: 0.0026042


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0026042, mid=0.0026042, abs_max=107.93389892578125
rel_dist={4: [-95.4365527136626, 95.43655271366259]}

## Binary search (step 7) starts
Candidate diff: 0.0013021


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0013021, mid=0.0013021, abs_max=107.93389892578125
rel_dist={4: [-95.43507596214961, 95.43507596214963]}

## Binary search (step 8) starts
Candidate diff: 0.0006510


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0006510, mid=0.0006510, abs_max=107.93389892578125
rel_dist={4: [-95.4342126817601, 95.4342126817601]}

## Binary search (step 9) starts
Candidate diff: 0.0003255


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0003255, mid=0.0003255, abs_max=107.93389892578125
rel_dist={4: [-95.43372888611276, 95.43372888611276]}

## Binary search (step 10) starts
Candidate diff: 0.0001628


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0001628, mid=0.0001628, abs_max=107.93389892578125
rel_dist={4: [-95.43347375507336, 95.43347375507335]}

## Binary search (step 11) starts
Candidate diff: 0.0000814


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000814, mid=0.0000814, abs_max=107.93389892578125
rel_dist={4: [-95.43334498914167, 95.43334498914169]}

## Binary search (step 12) starts
Candidate diff: 0.0000407


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000407, mid=0.0000407, abs_max=107.93389892578125
rel_dist={4: [-95.43328060632305, 95.43328060632305]}

## Binary search (step 13) starts
Candidate diff: 0.0000203


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000203, mid=0.0000203, abs_max=107.93389892578125
rel_dist={4: [-95.43324803845512, 95.43324803845513]}

## Binary search (step 14) starts
Candidate diff: 0.0000102


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000102, mid=0.0000102, abs_max=107.93389892578125
rel_dist={4: [-95.4332314428192, 95.43323144281925]}

## Binary search (step 15) starts
Candidate diff: 0.0000051


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000051, mid=0.0000051, abs_max=107.93389892578125
rel_dist={4: [-95.43322314584034, 95.43322314609429]}

## Binary search (step 16) starts
Candidate diff: 0.0000025


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000025, mid=0.0000025, abs_max=107.93389892578125
rel_dist={4: [-95.43321899120869, 95.43321899969155]}

## Binary search (step 17) starts
Candidate diff: 0.0000013


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000013, mid=0.0000013, abs_max=107.93389892578125
rel_dist={4: [-95.433216924984, 95.43321694589028]}

## Binary search (step 18) starts
Candidate diff: 0.0000006


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000006, mid=0.0000006, abs_max=107.93389892578125
rel_dist={4: [-95.43321594479565, 95.43321789875597]}

## Binary Search Result
Binary search time: 65.55 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1131.09 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4367909, upper bound: 95.4367909
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4367909, upper bound: 95.4367909
time: 0.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 4, lower bound: -95.4367909, upper bound: 95.4367909
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.17
Output dim: 4, lower bound: -95.4367909, upper bound: 95.4367909

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4172711, upper bound: 95.4192869
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4172711, upper bound: 95.4192869
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4261792, upper bound: 95.4261792
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4261792, upper bound: 95.4261792
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.79
Output dim: 4, lower bound: -95.4172711, upper bound: 95.4192869
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.79
Output dim: 4, lower bound: -95.4172711, upper bound: 95.4192869
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -95.4261792, upper bound: 95.4261792
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -95.4261792, upper bound: 95.4261792

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4240331, upper bound: 95.4240857
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233397, upper bound: 95.4240857
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4052966, upper bound: 95.4052966
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4052966, upper bound: 95.4052966
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.81 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4240331, upper bound: 95.4240857
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4233397, upper bound: 95.4240857
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4052966, upper bound: 95.4052966
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4052966, upper bound: 95.4052966

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3536456, upper bound: 95.3534734
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3536456, upper bound: 95.3534734
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.89 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.89
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 4, lower bound: -95.3536456, upper bound: 95.3534734
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.89
Output dim: 4, lower bound: -95.3536456, upper bound: 95.3534734

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4196445, upper bound: 95.4202729
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4171080, upper bound: 95.4170830
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4171552, upper bound: 95.4154845
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.87 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.87
Output dim: 4, lower bound: -95.4196445, upper bound: 95.4202729
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.87
Output dim: 4, lower bound: -95.4171080, upper bound: 95.4170830
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.87
Output dim: 4, lower bound: -95.4171552, upper bound: 95.4154845

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4185816, upper bound: 95.4151156
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4170907, upper bound: 95.4198817
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4190392, upper bound: 95.4169140
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4196900
time: 0.58 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.99 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.99
Output dim: 4, lower bound: -95.4185816, upper bound: 95.4151156
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 4, lower bound: -95.4170907, upper bound: 95.4198817
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.99
Output dim: 4, lower bound: -95.4190392, upper bound: 95.4169140
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.99
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4196900

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3781662, upper bound: 95.3775678
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3781662, upper bound: 95.3775678
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3783715
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3783715
time: 0.60 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.36 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.36
Output dim: 4, lower bound: -95.3781662, upper bound: 95.3775678
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.36
Output dim: 4, lower bound: -95.3781662, upper bound: 95.3775678
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.36
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3783715
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.36
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3783715
Binary search (step 0): status=Status.VERIFIED, low=0.1666667, high=0.3333333, mid=0.1666667, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.37
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
time: 0.60 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3958968, upper bound: 95.3948804
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3958968, upper bound: 95.3954685
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3985790, upper bound: 95.3967518
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3985790, upper bound: 95.3977574
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814147
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3819739
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3819739
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -95.3958968, upper bound: 95.3948804
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -95.3958968, upper bound: 95.3954685
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -95.3985790, upper bound: 95.3967518
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -95.3985790, upper bound: 95.3977574
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814147
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3819739
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3819739
Binary search (step 1): status=Status.VERIFIED, low=0.2500000, high=0.3333333, mid=0.2500000, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 2) starts
Candidate diff: 0.2916666


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4274503, upper bound: 95.4274503
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4274503, upper bound: 95.4274503
time: 0.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -95.4274503, upper bound: 95.4274503
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -95.4274503, upper bound: 95.4274503

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241999, upper bound: 95.4241999
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241999, upper bound: 95.4269394
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4269522, upper bound: 95.4269522
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4269522, upper bound: 95.4272576
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 4, lower bound: -95.4241999, upper bound: 95.4241999
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 4, lower bound: -95.4241999, upper bound: 95.4269394
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 4, lower bound: -95.4269522, upper bound: 95.4269522
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 4, lower bound: -95.4269522, upper bound: 95.4272576

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237702, upper bound: 95.4238169
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4259552, upper bound: 95.4237695
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4224857, upper bound: 95.4256211
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4224857, upper bound: 95.4256211
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4268177, upper bound: 95.4262081
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4261224, upper bound: 95.4263388
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4258450, upper bound: 95.4256180
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4256180, upper bound: 95.4256180
time: 0.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 4, lower bound: -95.4237702, upper bound: 95.4238169
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 4, lower bound: -95.4259552, upper bound: 95.4237695
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 4, lower bound: -95.4224857, upper bound: 95.4256211
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 4, lower bound: -95.4224857, upper bound: 95.4256211
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 4, lower bound: -95.4268177, upper bound: 95.4262081
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 4, lower bound: -95.4261224, upper bound: 95.4263388
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 4, lower bound: -95.4258450, upper bound: 95.4256180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.09
Output dim: 4, lower bound: -95.4256180, upper bound: 95.4256180

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4262973, upper bound: 95.4231592
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4257845, upper bound: 95.4236288
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3821419, upper bound: 95.3817962
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3821537, upper bound: 95.3815442
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4020520, upper bound: 95.4037253
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4020520, upper bound: 95.4037253
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3825294, upper bound: 95.3824322
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3824079, upper bound: 95.3824322
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4262973, upper bound: 95.4232032
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4235926, upper bound: 95.4256560
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4211755, upper bound: 95.4232133
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4230045, upper bound: 95.4230551
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237451, upper bound: 95.4230326
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237451, upper bound: 95.4234517
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4252865, upper bound: 95.4222177
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4201121, upper bound: 95.4250359
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4262973, upper bound: 95.4231592
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4257845, upper bound: 95.4236288
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.3821419, upper bound: 95.3817962
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.3821537, upper bound: 95.3815442
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4020520, upper bound: 95.4037253
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4020520, upper bound: 95.4037253
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.3825294, upper bound: 95.3824322
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.3824079, upper bound: 95.3824322
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4262973, upper bound: 95.4232032
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4235926, upper bound: 95.4256560
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4211755, upper bound: 95.4232133
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4230045, upper bound: 95.4230551
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4237451, upper bound: 95.4230326
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4237451, upper bound: 95.4234517
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4252865, upper bound: 95.4222177
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.98
Output dim: 4, lower bound: -95.4201121, upper bound: 95.4250359

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3792679, upper bound: 95.3786814
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3792679, upper bound: 95.3786814
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4225315, upper bound: 95.4206791
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4226839, upper bound: 95.4206392
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4234757, upper bound: 95.4201742
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4234756, upper bound: 95.4200881
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4199941, upper bound: 95.4217561
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4199941, upper bound: 95.4230527
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206711, upper bound: 95.4200911
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206392, upper bound: 95.4226839
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4202729, upper bound: 95.4196445
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4202729, upper bound: 95.4196445
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4159435, upper bound: 95.4197842
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4204453, upper bound: 95.4179951
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3794507, upper bound: 95.3800573
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3794507, upper bound: 95.3800573
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795148
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795148
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795046
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795046
time: 0.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.3792679, upper bound: 95.3786814
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.3792679, upper bound: 95.3786814
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4225315, upper bound: 95.4206791
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4226839, upper bound: 95.4206392
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4234757, upper bound: 95.4201742
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4234756, upper bound: 95.4200881
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4199941, upper bound: 95.4217561
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4199941, upper bound: 95.4230527
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4206711, upper bound: 95.4200911
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4206392, upper bound: 95.4226839
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4202729, upper bound: 95.4196445
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4202729, upper bound: 95.4196445
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4159435, upper bound: 95.4197842
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.4204453, upper bound: 95.4179951
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.3794507, upper bound: 95.3800573
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.3794507, upper bound: 95.3800573
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795148
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795148
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795046
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795046

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4190392, upper bound: 95.4169140
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4190392, upper bound: 95.4169140
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4198342, upper bound: 95.4168621
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4168605
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3780625, upper bound: 95.3773446
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3780625, upper bound: 95.3773446
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4196900, upper bound: 95.4151156
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4196900, upper bound: 95.4165283
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4168605, upper bound: 95.4185947
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4168853, upper bound: 95.4152525
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3782266, upper bound: 95.3782266
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3784914, upper bound: 95.3782266
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3776193, upper bound: 95.3773446
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4168605, upper bound: 95.4198342
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4168621, upper bound: 95.4198342
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3791415, upper bound: 95.3781142
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3791415, upper bound: 95.3781142
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3790772, upper bound: 95.3781142
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3790772, upper bound: 95.3781142
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3781142, upper bound: 95.3791577
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3781142, upper bound: 95.3791577
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4198817, upper bound: 95.4170907
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4171587
time: 0.60 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4190392, upper bound: 95.4169140
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4190392, upper bound: 95.4169140
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4198342, upper bound: 95.4168621
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4168605
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3780625, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3780625, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4196900, upper bound: 95.4151156
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4196900, upper bound: 95.4165283
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4168605, upper bound: 95.4185947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4168853, upper bound: 95.4152525
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3782266, upper bound: 95.3782266
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3784914, upper bound: 95.3782266
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3776193, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4168605, upper bound: 95.4198342
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4168621, upper bound: 95.4198342
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3791415, upper bound: 95.3781142
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3791415, upper bound: 95.3781142
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3790772, upper bound: 95.3781142
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3790772, upper bound: 95.3781142
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3781142, upper bound: 95.3791577
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.3781142, upper bound: 95.3791577
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4198817, upper bound: 95.4170907
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.95
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4171587

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3775385
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3782243, upper bound: 95.3773446
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3782243, upper bound: 95.3773446
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3782716, upper bound: 95.3773446
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3782716, upper bound: 95.3773446
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775027, upper bound: 95.3773446
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775385, upper bound: 95.3773446
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3782421
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3782421
time: 0.58 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3775385
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3782243, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3782243, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3782716, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3782716, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3775027, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3775385, upper bound: 95.3773446
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3782421
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.65
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3782421
Binary search (step 2): status=Status.VERIFIED, low=0.2916666, high=0.3333333, mid=0.2916666, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 3) starts
Candidate diff: 0.3125000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4374201
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4385598
time: 0.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4374201
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.24
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4385598

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4251220, upper bound: 95.4237285
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4251220, upper bound: 95.4237285
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4251220
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4251220
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 4, lower bound: -95.4251220, upper bound: 95.4237285
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 4, lower bound: -95.4251220, upper bound: 95.4237285
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4251220
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.86
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4251220

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4189668, upper bound: 95.4177960
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4189668, upper bound: 95.4176512
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4006621, upper bound: 95.4006621
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4006621, upper bound: 95.4006621
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4235136, upper bound: 95.4250178
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4236584, upper bound: 95.4220691
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4051176
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4051176
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 4, lower bound: -95.4189668, upper bound: 95.4177960
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 4, lower bound: -95.4189668, upper bound: 95.4176512
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 4, lower bound: -95.4006621, upper bound: 95.4006621
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 4, lower bound: -95.4006621, upper bound: 95.4006621
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 4, lower bound: -95.4235136, upper bound: 95.4250178
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 4, lower bound: -95.4236584, upper bound: 95.4220691
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4051176
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4051176

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4231332, upper bound: 95.4234165
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4223796, upper bound: 95.4246499
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3482239, upper bound: 95.3482239
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3482239, upper bound: 95.3482239
time: 0.53 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.09 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.09
Output dim: 4, lower bound: -95.4231332, upper bound: 95.4234165
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.09
Output dim: 4, lower bound: -95.4223796, upper bound: 95.4246499
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 4, lower bound: -95.3482239, upper bound: 95.3482239
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.09
Output dim: 4, lower bound: -95.3482239, upper bound: 95.3482239

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4158586, upper bound: 95.4159463
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4153054, upper bound: 95.4159463
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207364, upper bound: 95.4245791
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207364, upper bound: 95.4233030
time: 0.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.86 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 4, lower bound: -95.4158586, upper bound: 95.4159463
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 4, lower bound: -95.4153054, upper bound: 95.4159463
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 4, lower bound: -95.4207364, upper bound: 95.4245791
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 4, lower bound: -95.4207364, upper bound: 95.4233030

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4139822, upper bound: 95.4175272
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4139822, upper bound: 95.4175272
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4157740, upper bound: 95.4173317
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4157457, upper bound: 95.4159297
time: 0.58 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.80 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 4, lower bound: -95.4139822, upper bound: 95.4175272
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 4, lower bound: -95.4139822, upper bound: 95.4175272
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 4, lower bound: -95.4157740, upper bound: 95.4173317
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.80
Output dim: 4, lower bound: -95.4157457, upper bound: 95.4159297
Binary search (step 3): status=Status.VERIFIED, low=0.3125000, high=0.3333333, mid=0.3125000, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 4) starts
Candidate diff: 0.3229166


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.27 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.27
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
time: 0.52 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4263094, upper bound: 95.4262864
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4263094, upper bound: 95.4272446
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3818832
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3818832
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814147
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814147
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4200404
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.89 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.4263094, upper bound: 95.4262864
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.4263094, upper bound: 95.4272446
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3818832
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3818832
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814147
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814147
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3781142, upper bound: 95.3792069
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3781142, upper bound: 95.3792069
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819739
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819739
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4172220, upper bound: 95.4173134
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4172220, upper bound: 95.4194375
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.16
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 4, lower bound: -95.3781142, upper bound: 95.3792069
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 4, lower bound: -95.3781142, upper bound: 95.3792069
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819739
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819739
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 4, lower bound: -95.4172220, upper bound: 95.4173134
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.16
Output dim: 4, lower bound: -95.4172220, upper bound: 95.4194375

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4159435, upper bound: 95.4192360
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4159435, upper bound: 95.4203399
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.94
Output dim: 4, lower bound: -95.4159435, upper bound: 95.4192360
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 4, lower bound: -95.4192469, upper bound: 95.4204453
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.94
Output dim: 4, lower bound: -95.4159435, upper bound: 95.4203399

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3789688, upper bound: 95.3781142
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3789688, upper bound: 95.3781142
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4185816, upper bound: 95.4151156
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4170907, upper bound: 95.4198817
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4164066, upper bound: 95.4163972
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4168621, upper bound: 95.4197736
time: 0.58 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -95.3789688, upper bound: 95.3781142
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -95.3789688, upper bound: 95.3781142
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -95.4185816, upper bound: 95.4151156
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -95.4170907, upper bound: 95.4198817
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -95.4164066, upper bound: 95.4163972
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -95.4168621, upper bound: 95.4197736

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3781662, upper bound: 95.3775678
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3781662, upper bound: 95.3775678
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775511, upper bound: 95.3773446
time: 0.66 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 4, lower bound: -95.3781662, upper bound: 95.3775678
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 4, lower bound: -95.3781662, upper bound: 95.3775678
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.01
Output dim: 4, lower bound: -95.3775511, upper bound: 95.3773446
Binary search (step 4): status=Status.VERIFIED, low=0.3229166, high=0.3333333, mid=0.3229166, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 5) starts
Candidate diff: 0.3281250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4381946, upper bound: 95.4381946
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4381946, upper bound: 95.4385552
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 4, lower bound: -95.4381946, upper bound: 95.4381946
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.09
Output dim: 4, lower bound: -95.4381946, upper bound: 95.4385552

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4180357, upper bound: 95.4164728
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4180357, upper bound: 95.4164728
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4307747, upper bound: 95.4314222
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4307747, upper bound: 95.4314222
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.75
Output dim: 4, lower bound: -95.4180357, upper bound: 95.4164728
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.75
Output dim: 4, lower bound: -95.4180357, upper bound: 95.4164728
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 4, lower bound: -95.4307747, upper bound: 95.4314222
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 4, lower bound: -95.4307747, upper bound: 95.4314222

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3789926, upper bound: 95.3801398
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3789926, upper bound: 95.3801398
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3794507, upper bound: 95.3800573
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3794507, upper bound: 95.3800573
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.63 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.63
Output dim: 4, lower bound: -95.3789926, upper bound: 95.3801398
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.63
Output dim: 4, lower bound: -95.3789926, upper bound: 95.3801398
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.63
Output dim: 4, lower bound: -95.3794507, upper bound: 95.3800573
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.63
Output dim: 4, lower bound: -95.3794507, upper bound: 95.3800573
Binary search (step 5): status=Status.VERIFIED, low=0.3281250, high=0.3333333, mid=0.3281250, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405608688]}

## Binary search (step 6) starts
Candidate diff: 0.3307291


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4194846, upper bound: 95.4194846
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4194846, upper bound: 95.4194846
time: 0.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 4, lower bound: -95.4194846, upper bound: 95.4194846
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.23
Output dim: 4, lower bound: -95.4194846, upper bound: 95.4194846

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4194796, upper bound: 95.4189962
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4189727, upper bound: 95.4194796
time: 0.59 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4172633, upper bound: 95.4168283
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4172633, upper bound: 95.4168283
time: 0.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -95.4194796, upper bound: 95.4189962
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.54
Output dim: 4, lower bound: -95.4189727, upper bound: 95.4194796
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.54
Output dim: 4, lower bound: -95.4172633, upper bound: 95.4168283
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.54
Output dim: 4, lower bound: -95.4172633, upper bound: 95.4168283

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4168079, upper bound: 95.4172583
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4164006, upper bound: 95.4172583
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3141126, upper bound: 95.3141126
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3141126, upper bound: 95.3141126
time: 0.54 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -95.4168079, upper bound: 95.4172583
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -95.4164006, upper bound: 95.4172583
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -95.3141126, upper bound: 95.3141126
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 4, lower bound: -95.3141126, upper bound: 95.3141126
Binary search (step 6): status=Status.VERIFIED, low=0.3307291, high=0.3333333, mid=0.3307291, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 7) starts
Candidate diff: 0.3320312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3945261, upper bound: 95.3945261
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3945261, upper bound: 95.3945261
time: 0.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.29 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.29
Output dim: 4, lower bound: -95.3945261, upper bound: 95.3945261
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.29
Output dim: 4, lower bound: -95.3945261, upper bound: 95.3945261
Binary search (step 7): status=Status.VERIFIED, low=0.3320312, high=0.3333333, mid=0.3320312, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 8) starts
Candidate diff: 0.3326823


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.13 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.13
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.13
Output dim: 4, lower bound: -95.4193349, upper bound: 95.4193349
Binary search (step 8): status=Status.VERIFIED, low=0.3326823, high=0.3333333, mid=0.3326823, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 9) starts
Candidate diff: 0.3330078


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.17 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.17
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.17
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
Binary search (step 9): status=Status.VERIFIED, low=0.3330078, high=0.3333333, mid=0.3330078, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 10) starts
Candidate diff: 0.3331706


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4367909, upper bound: 95.4367909
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4367909, upper bound: 95.4367909
time: 0.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.19 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 4, lower bound: -95.4367909, upper bound: 95.4367909
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 4, lower bound: -95.4367909, upper bound: 95.4367909

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4194491, upper bound: 95.4190507
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4194491, upper bound: 95.4190296
time: 0.54 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4252963, upper bound: 95.4237958
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237958, upper bound: 95.4238428
time: 0.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.06
Output dim: 4, lower bound: -95.4194491, upper bound: 95.4190507
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.06
Output dim: 4, lower bound: -95.4194491, upper bound: 95.4190296
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 4, lower bound: -95.4252963, upper bound: 95.4237958
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.06
Output dim: 4, lower bound: -95.4237958, upper bound: 95.4238428

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4235103, upper bound: 95.4236558
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4235103, upper bound: 95.4236974
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3674395, upper bound: 95.3673636
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3674395, upper bound: 95.3673636
time: 0.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.89 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.4235103, upper bound: 95.4236558
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.4235103, upper bound: 95.4236974
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.3674395, upper bound: 95.3673636
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 4, lower bound: -95.3674395, upper bound: 95.3673636

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3665580, upper bound: 95.3666482
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3665580, upper bound: 95.3666482
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207529, upper bound: 95.4229853
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207530, upper bound: 95.4233410
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.84 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -95.3665580, upper bound: 95.3666482
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -95.3665580, upper bound: 95.3666482
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -95.4207529, upper bound: 95.4229853
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -95.4207530, upper bound: 95.4233410

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4166961, upper bound: 95.4157992
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4172597, upper bound: 95.4157992
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3921719, upper bound: 95.3923690
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3921719, upper bound: 95.3923690
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.80 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 4, lower bound: -95.4166961, upper bound: 95.4157992
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 4, lower bound: -95.4172597, upper bound: 95.4157992
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 4, lower bound: -95.3921719, upper bound: 95.3923690
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.80
Output dim: 4, lower bound: -95.3921719, upper bound: 95.3923690
Binary search (step 10): status=Status.VERIFIED, low=0.3331706, high=0.3333333, mid=0.3331706, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 11) starts
Candidate diff: 0.3332519


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4374201
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4385598
time: 0.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4374201
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4385598

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4372734, upper bound: 95.4372734
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4372734, upper bound: 95.4373598
time: 1.00 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4329043, upper bound: 95.4344565
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4329043, upper bound: 95.4344710
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -95.4372734, upper bound: 95.4372734
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -95.4372734, upper bound: 95.4373598
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -95.4329043, upper bound: 95.4344565
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 4, lower bound: -95.4329043, upper bound: 95.4344710

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4225961, upper bound: 95.4225961
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4247553, upper bound: 95.4225961
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3790424, upper bound: 95.3795148
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3790424, upper bound: 95.3795148
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4327134, upper bound: 95.4338307
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4327134, upper bound: 95.4344532
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4290232, upper bound: 95.4314105
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4289245, upper bound: 95.4314105
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -95.4225961, upper bound: 95.4225961
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -95.4247553, upper bound: 95.4225961
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 4, lower bound: -95.3790424, upper bound: 95.3795148
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 4, lower bound: -95.3790424, upper bound: 95.3795148
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -95.4327134, upper bound: 95.4338307
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -95.4327134, upper bound: 95.4344532
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -95.4290232, upper bound: 95.4314105
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 4, lower bound: -95.4289245, upper bound: 95.4314105

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3923690, upper bound: 95.3921719
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3923690, upper bound: 95.3921719
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4216957, upper bound: 95.4225600
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4246499, upper bound: 95.4223796
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4061829, upper bound: 95.4069322
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4061829, upper bound: 95.4069322
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3921719, upper bound: 95.3923690
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3921719, upper bound: 95.3923690
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3809674, upper bound: 95.3812960
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3809674, upper bound: 95.3812960
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4191872, upper bound: 95.4187057
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4191872, upper bound: 95.4212922
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.3923690, upper bound: 95.3921719
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.3923690, upper bound: 95.3921719
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.4216957, upper bound: 95.4225600
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.4246499, upper bound: 95.4223796
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.4061829, upper bound: 95.4069322
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.4061829, upper bound: 95.4069322
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.3921719, upper bound: 95.3923690
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.3921719, upper bound: 95.3923690
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.3809674, upper bound: 95.3812960
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.3809674, upper bound: 95.3812960
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.4191872, upper bound: 95.4187057
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -95.4191872, upper bound: 95.4212922

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4162404, upper bound: 95.4166495
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4162404, upper bound: 95.4166272
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4158107, upper bound: 95.4169453
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4173330, upper bound: 95.4169453
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3928240, upper bound: 95.3926686
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3928317, upper bound: 95.3926405
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 4, lower bound: -95.4162404, upper bound: 95.4166495
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 4, lower bound: -95.4162404, upper bound: 95.4166272
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 4, lower bound: -95.4158107, upper bound: 95.4169453
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 4, lower bound: -95.4173330, upper bound: 95.4169453
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 4, lower bound: -95.3928240, upper bound: 95.3926686
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.08
Output dim: 4, lower bound: -95.3928317, upper bound: 95.3926405
Binary search (step 11): status=Status.VERIFIED, low=0.3332519, high=0.3333333, mid=0.3332519, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 12) starts
Candidate diff: 0.3332926


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3945261, upper bound: 95.3945261
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3945261, upper bound: 95.3945261
time: 0.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.29 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.29
Output dim: 4, lower bound: -95.3945261, upper bound: 95.3945261
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.29
Output dim: 4, lower bound: -95.3945261, upper bound: 95.3945261
Binary search (step 12): status=Status.VERIFIED, low=0.3332926, high=0.3333333, mid=0.3332926, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 13) starts
Candidate diff: 0.3333130


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4374201
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4385598
time: 0.59 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4374201
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.16
Output dim: 4, lower bound: -95.4374201, upper bound: 95.4385598

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4237285
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4237285
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241999, upper bound: 95.4269394
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241999, upper bound: 95.4269394
time: 0.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4237285
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4237285
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 4, lower bound: -95.4241999, upper bound: 95.4269394
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.73
Output dim: 4, lower bound: -95.4241999, upper bound: 95.4269394

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4235136, upper bound: 95.4236584
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4236584, upper bound: 95.4221499
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4027999
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4051176, upper bound: 95.4027999
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3578218, upper bound: 95.3578794
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3578218, upper bound: 95.3578794
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237246, upper bound: 95.4264138
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237246, upper bound: 95.4267446
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4235136, upper bound: 95.4236584
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4236584, upper bound: 95.4221499
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4027999
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4051176, upper bound: 95.4027999
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.3578218, upper bound: 95.3578794
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.3578218, upper bound: 95.3578794
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4237246, upper bound: 95.4264138
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4237246, upper bound: 95.4267446

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237257, upper bound: 95.4235257
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245438, upper bound: 95.4235774
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3924972, upper bound: 95.3911157
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3924972, upper bound: 95.3911157
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4222177, upper bound: 95.4252865
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4222177, upper bound: 95.4252865
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4214809, upper bound: 95.4256494
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4231592, upper bound: 95.4262973
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 4, lower bound: -95.4237257, upper bound: 95.4235257
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 4, lower bound: -95.4245438, upper bound: 95.4235774
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 4, lower bound: -95.3924972, upper bound: 95.3911157
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.86
Output dim: 4, lower bound: -95.3924972, upper bound: 95.3911157
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 4, lower bound: -95.4222177, upper bound: 95.4252865
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 4, lower bound: -95.4222177, upper bound: 95.4252865
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 4, lower bound: -95.4214809, upper bound: 95.4256494
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.86
Output dim: 4, lower bound: -95.4231592, upper bound: 95.4262973

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3918707, upper bound: 95.3930224
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3918707, upper bound: 95.3930224
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3482239, upper bound: 95.3482239
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3482239, upper bound: 95.3482239
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4188946, upper bound: 95.4217613
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4189444, upper bound: 95.4209173
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3784804
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3787345, upper bound: 95.3784804
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4181695, upper bound: 95.4223898
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4181695, upper bound: 95.4226909
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3782266, upper bound: 95.3792679
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3786814, upper bound: 95.3792679
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.3918707, upper bound: 95.3930224
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.3918707, upper bound: 95.3930224
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.3482239, upper bound: 95.3482239
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.3482239, upper bound: 95.3482239
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.4188946, upper bound: 95.4217613
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.4189444, upper bound: 95.4209173
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3784804
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.3787345, upper bound: 95.3784804
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.4181695, upper bound: 95.4223898
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.4181695, upper bound: 95.4226909
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.3782266, upper bound: 95.3792679
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.63
Output dim: 4, lower bound: -95.3786814, upper bound: 95.3792679

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4168605, upper bound: 95.4185947
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4168605, upper bound: 95.4198342
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3785660, upper bound: 95.3775864
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3785660, upper bound: 95.3775864
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3782266, upper bound: 95.3793133
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3782266, upper bound: 95.3793133
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4194464
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4171587
time: 0.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.87 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.87
Output dim: 4, lower bound: -95.4168605, upper bound: 95.4185947
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.87
Output dim: 4, lower bound: -95.4168605, upper bound: 95.4198342
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.87
Output dim: 4, lower bound: -95.3785660, upper bound: 95.3775864
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.87
Output dim: 4, lower bound: -95.3785660, upper bound: 95.3775864
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.87
Output dim: 4, lower bound: -95.3782266, upper bound: 95.3793133
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.87
Output dim: 4, lower bound: -95.3782266, upper bound: 95.3793133
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.87
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4194464
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.87
Output dim: 4, lower bound: -95.4151156, upper bound: 95.4171587

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775027, upper bound: 95.3773446
time: 0.62 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.73 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 4, lower bound: -95.3773446, upper bound: 95.3773446
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.73
Output dim: 4, lower bound: -95.3775027, upper bound: 95.3773446
Binary search (step 13): status=Status.VERIFIED, low=0.3333130, high=0.3333333, mid=0.3333130, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 14) starts
Candidate diff: 0.3333231


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4298016, upper bound: 95.4298016
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4298016, upper bound: 95.4298016
time: 0.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 4, lower bound: -95.4298016, upper bound: 95.4298016
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 4, lower bound: -95.4298016, upper bound: 95.4298016

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4289798, upper bound: 95.4289798
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4289798, upper bound: 95.4295060
time: 0.57 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4290134, upper bound: 95.4297243
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4297243, upper bound: 95.4290800
time: 0.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 4, lower bound: -95.4289798, upper bound: 95.4289798
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 4, lower bound: -95.4289798, upper bound: 95.4295060
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 4, lower bound: -95.4290134, upper bound: 95.4297243
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 4, lower bound: -95.4297243, upper bound: 95.4290800

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4292895, upper bound: 95.4273772
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273772, upper bound: 95.4287432
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4184721, upper bound: 95.4185266
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4184721, upper bound: 95.4185031
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4032307, upper bound: 95.4043267
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4032307, upper bound: 95.4043267
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3944177, upper bound: 95.3930430
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3944177, upper bound: 95.3930430
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 4, lower bound: -95.4292895, upper bound: 95.4273772
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 4, lower bound: -95.4273772, upper bound: 95.4287432
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 4, lower bound: -95.4184721, upper bound: 95.4185266
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 4, lower bound: -95.4184721, upper bound: 95.4185031
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 4, lower bound: -95.4032307, upper bound: 95.4043267
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 4, lower bound: -95.4032307, upper bound: 95.4043267
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 4, lower bound: -95.3944177, upper bound: 95.3930430
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 4, lower bound: -95.3944177, upper bound: 95.3930430

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4171356, upper bound: 95.4155001
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4158510, upper bound: 95.4155001
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4276460, upper bound: 95.4286707
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4265362, upper bound: 95.4284880
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 4, lower bound: -95.4171356, upper bound: 95.4155001
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.21
Output dim: 4, lower bound: -95.4158510, upper bound: 95.4155001
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 4, lower bound: -95.4276460, upper bound: 95.4286707
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.21
Output dim: 4, lower bound: -95.4265362, upper bound: 95.4284880

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4161554, upper bound: 95.4162669
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4157513, upper bound: 95.4162669
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4165839, upper bound: 95.4165639
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4163445, upper bound: 95.4159084
time: 0.72 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.38
Output dim: 4, lower bound: -95.4161554, upper bound: 95.4162669
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.38
Output dim: 4, lower bound: -95.4157513, upper bound: 95.4162669
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.38
Output dim: 4, lower bound: -95.4165839, upper bound: 95.4165639
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.38
Output dim: 4, lower bound: -95.4163445, upper bound: 95.4159084
Binary search (step 14): status=Status.VERIFIED, low=0.3333231, high=0.3333333, mid=0.3333231, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 15) starts
Candidate diff: 0.3333282


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.17 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.17
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.17
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
Binary search (step 15): status=Status.VERIFIED, low=0.3333282, high=0.3333333, mid=0.3333282, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 16) starts
Candidate diff: 0.3333308


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.17 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.17
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.17
Output dim: 4, lower bound: -95.3835054, upper bound: 95.3835054
Binary search (step 16): status=Status.VERIFIED, low=0.3333308, high=0.3333333, mid=0.3333308, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 17) starts
Candidate diff: 0.3333320


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4253754, upper bound: 95.4253754
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4253754, upper bound: 95.4253754
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -95.4253754, upper bound: 95.4253754
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.13
Output dim: 4, lower bound: -95.4253754, upper bound: 95.4253754

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4237285
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4251220
time: 0.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4022143, upper bound: 95.4022143
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4022143, upper bound: 95.4022143
time: 0.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4237285
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.12
Output dim: 4, lower bound: -95.4237285, upper bound: 95.4251220
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.12
Output dim: 4, lower bound: -95.4022143, upper bound: 95.4022143
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.12
Output dim: 4, lower bound: -95.4022143, upper bound: 95.4022143

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4027999
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4027999
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4167125, upper bound: 95.4189933
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4166734, upper bound: 95.4189933
time: 0.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4027999
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4027999, upper bound: 95.4027999
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4167125, upper bound: 95.4189933
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4166734, upper bound: 95.4189933
Binary search (step 17): status=Status.VERIFIED, low=0.3333320, high=0.3333333, mid=0.3333320, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 18) starts
Candidate diff: 0.3333327


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4381946, upper bound: 95.4381946
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4381946, upper bound: 95.4385552
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.11 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 4, lower bound: -95.4381946, upper bound: 95.4381946
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.11
Output dim: 4, lower bound: -95.4381946, upper bound: 95.4385552

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4289798, upper bound: 95.4289798
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4289798, upper bound: 95.4289798
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4372734, upper bound: 95.4373598
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4372734, upper bound: 95.4385530
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 4, lower bound: -95.4289798, upper bound: 95.4289798
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 4, lower bound: -95.4289798, upper bound: 95.4289798
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 4, lower bound: -95.4372734, upper bound: 95.4373598
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.27
Output dim: 4, lower bound: -95.4372734, upper bound: 95.4385530

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4073199, upper bound: 95.4074276
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4073199, upper bound: 95.4074276
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4180335, upper bound: 95.4145133
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4180335, upper bound: 95.4159150
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273772, upper bound: 95.4277199
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4287432, upper bound: 95.4277199
time: 0.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237246, upper bound: 95.4267446
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4237246, upper bound: 95.4267446
time: 0.60 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4073199, upper bound: 95.4074276
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4073199, upper bound: 95.4074276
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4180335, upper bound: 95.4145133
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4180335, upper bound: 95.4159150
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4273772, upper bound: 95.4277199
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4287432, upper bound: 95.4277199
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4237246, upper bound: 95.4267446
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 4, lower bound: -95.4237246, upper bound: 95.4267446

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4158257, upper bound: 95.4166370
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4143228, upper bound: 95.4166370
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4159167, upper bound: 95.4180445
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4165639, upper bound: 95.4180445
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4205623, upper bound: 95.4238830
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206207, upper bound: 95.4239007
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4219095, upper bound: 95.4250359
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4201121, upper bound: 95.4250359
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.63 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 4, lower bound: -95.4158257, upper bound: 95.4166370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 4, lower bound: -95.4143228, upper bound: 95.4166370
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 4, lower bound: -95.4159167, upper bound: 95.4180445
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.63
Output dim: 4, lower bound: -95.4165639, upper bound: 95.4180445
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 4, lower bound: -95.4205623, upper bound: 95.4238830
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 4, lower bound: -95.4206207, upper bound: 95.4239007
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 4, lower bound: -95.4219095, upper bound: 95.4250359
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 4, lower bound: -95.4201121, upper bound: 95.4250359

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3785575
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3785575
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4168815, upper bound: 95.4189319
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4168815, upper bound: 95.4184268
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4168815, upper bound: 95.4215416
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4184663, upper bound: 95.4204626
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795046
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795046
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.89 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3785575
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3785575
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 4, lower bound: -95.4168815, upper bound: 95.4189319
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 4, lower bound: -95.4168815, upper bound: 95.4184268
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.89
Output dim: 4, lower bound: -95.4168815, upper bound: 95.4215416
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.89
Output dim: 4, lower bound: -95.4184663, upper bound: 95.4204626
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795046
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.89
Output dim: 4, lower bound: -95.3784804, upper bound: 95.3795046

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3785511
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3785511
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3784140
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3779124, upper bound: 95.3784140
time: 0.76 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.37 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3785511
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3785511
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 4, lower bound: -95.3775864, upper bound: 95.3784140
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 4, lower bound: -95.3779124, upper bound: 95.3784140
Binary search (step 18): status=Status.VERIFIED, low=0.3333327, high=0.3333333, mid=0.3333327, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.3333326776822787
execution time: 628.36 seconds
