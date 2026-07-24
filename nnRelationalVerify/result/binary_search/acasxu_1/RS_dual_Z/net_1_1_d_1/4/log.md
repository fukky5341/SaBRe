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
execution time: IAR + LP analysis = 1.70 + 1.62 = 3.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -95.4385641, upper bound: 95.4385641


# Binary Search by BASE starts (time budget: 1196.69 seconds, max iter: 100)

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
Binary search time: 65.57 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1131.11 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461
time: 0.55 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
time: 0.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4272223
time: 0.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4272223

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4213516
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241638
time: 0.62 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.07 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207666
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179338
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3814728
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814728
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814371
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814371
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3807267, upper bound: 95.3814256
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3810313, upper bound: 95.3814256
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3818832
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3803924, upper bound: 95.3815360
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3809535, upper bound: 95.3815360
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3805195, upper bound: 95.3814376
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3809812, upper bound: 95.3814376
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3809812
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3805195
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3809535
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3803924
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814147
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3814378
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3814378
time: 0.70 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.05 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3814728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814371
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814371
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3807267, upper bound: 95.3814256
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3810313, upper bound: 95.3814256
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3818832
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3803924, upper bound: 95.3815360
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3809535, upper bound: 95.3815360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3805195, upper bound: 95.3814376
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3809812, upper bound: 95.3814376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3809812
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3805195
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3809535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3803924
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814147
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3814378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3814378
Binary search (step 0): status=Status.VERIFIED, low=0.1666667, high=0.3333333, mid=0.1666667, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

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
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
time: 0.60 seconds

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

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245467, upper bound: 95.4245671
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245467, upper bound: 95.4245671
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 4, lower bound: -95.4245467, upper bound: 95.4245671
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.14
Output dim: 4, lower bound: -95.4245467, upper bound: 95.4245671

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241638
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.63 seconds

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

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179338
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814728
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814728
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3819318, upper bound: 95.3814371
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3819318, upper bound: 95.3814371
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3818832
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3818832
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3807267, upper bound: 95.3814256
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3810313, upper bound: 95.3814256
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3803924, upper bound: 95.3815360
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3809535, upper bound: 95.3815360
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3805195, upper bound: 95.3814376
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3809812, upper bound: 95.3814376
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3809535
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3803924
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814300, upper bound: 95.3819318
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814378
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814378
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3819318, upper bound: 95.3814371
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3819318, upper bound: 95.3814371
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3818832
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3818832
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3807267, upper bound: 95.3814256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3810313, upper bound: 95.3814256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3803924, upper bound: 95.3815360
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3809535, upper bound: 95.3815360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3805195, upper bound: 95.3814376
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3809812, upper bound: 95.3814376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3809535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3803924
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814300, upper bound: 95.3819318
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.89
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814378
Binary search (step 1): status=Status.VERIFIED, low=0.2500000, high=0.3333333, mid=0.2500000, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 2) starts
Candidate diff: 0.2916666


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

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
time: 0.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
time: 0.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245467, upper bound: 95.4245671
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4245467, upper bound: 95.4245671
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 4, lower bound: -95.4245671, upper bound: 95.4245467
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 4, lower bound: -95.4245467, upper bound: 95.4245671
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.96
Output dim: 4, lower bound: -95.4245467, upper bound: 95.4245671

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
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
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.59 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3814728
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814728
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3819318, upper bound: 95.3814371
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814371
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3818832
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3818832
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3807267, upper bound: 95.3814256
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3810313, upper bound: 95.3814256
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3819739, upper bound: 95.3814300
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3819739, upper bound: 95.3814300
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3803924, upper bound: 95.3815360
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3809535, upper bound: 95.3815360
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3805195, upper bound: 95.3814376
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3809812, upper bound: 95.3814376
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3809535
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3803924
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3810313
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3807267
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814378
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3814378
time: 0.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3814728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3819318, upper bound: 95.3814371
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814378, upper bound: 95.3814371
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3818832
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814147, upper bound: 95.3818832
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814031, upper bound: 95.3814373
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3807267, upper bound: 95.3814256
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3810313, upper bound: 95.3814256
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3819739, upper bound: 95.3814300
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3819739, upper bound: 95.3814300
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3803924, upper bound: 95.3815360
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3809535, upper bound: 95.3815360
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3805195, upper bound: 95.3814376
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3809812, upper bound: 95.3814376
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3809535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814376, upper bound: 95.3803924
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3810313
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814256, upper bound: 95.3807267
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814373, upper bound: 95.3814031
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3818832, upper bound: 95.3814147
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3819318
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814728, upper bound: 95.3814378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.91
Output dim: 4, lower bound: -95.3814371, upper bound: 95.3814378
Binary search (step 2): status=Status.VERIFIED, low=0.2916666, high=0.3333333, mid=0.2916666, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 3) starts
Candidate diff: 0.3125000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

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
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

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

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.64 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.05
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.05
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.05
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.05
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207666
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4206732
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.04
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.95
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
Binary search (step 3): status=Status.VERIFIED, low=0.3125000, high=0.3333333, mid=0.3125000, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 4) starts
Candidate diff: 0.3229166


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

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
- Time for RS candidates: 1.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

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

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.95
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.57 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.64
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
Binary search (step 4): status=Status.VERIFIED, low=0.3229166, high=0.3333333, mid=0.3229166, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 5) starts
Candidate diff: 0.3281250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

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
time: 0.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207666
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179370
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4200404
time: 0.63 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.15
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.50
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
Binary search (step 5): status=Status.VERIFIED, low=0.3281250, high=0.3333333, mid=0.3281250, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405608688]}

## Binary search (step 6) starts
Candidate diff: 0.3307291


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.43 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.43
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
time: 0.53 seconds

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

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4271864
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4233232
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4213450
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241638
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.20
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.45
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
Binary search (step 6): status=Status.VERIFIED, low=0.3307291, high=0.3333333, mid=0.3307291, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 7) starts
Candidate diff: 0.3320312


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.44 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

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

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.01
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4233232
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4213450
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.95
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.59 seconds

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

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4206732, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.01
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.53
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
Binary search (step 7): status=Status.VERIFIED, low=0.3320312, high=0.3333333, mid=0.3320312, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 8) starts
Candidate diff: 0.3326823


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
time: 0.53 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.55 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.87 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.87
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4233232
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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4213450
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179370
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4200404
time: 0.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4200781, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
Binary search (step 8): status=Status.VERIFIED, low=0.3326823, high=0.3333333, mid=0.3326823, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 9) starts
Candidate diff: 0.3330078


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

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
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
time: 0.55 seconds

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

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

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
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4233232
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4213450
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241638
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4213516, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179370
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179338
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
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4200404
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4207260, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.02
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483912, upper bound: 95.3483745
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.44
Output dim: 4, lower bound: -95.3483756, upper bound: 95.3485283
Binary search (step 9): status=Status.VERIFIED, low=0.3330078, high=0.3333333, mid=0.3330078, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary search (step 10) starts
Candidate diff: 0.3331706


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345320
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 4, lower bound: -95.4345320, upper bound: 95.4345461

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
time: 0.55 seconds

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 4, lower bound: -95.4272223, upper bound: 95.4273401
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 4, lower bound: -95.4271864, upper bound: 95.4273401
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4271864
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 4, lower bound: -95.4273401, upper bound: 95.4272223

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4213450
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241638
time: 0.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 4, lower bound: -95.4241638, upper bound: 95.4233232
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241643
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4213450
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 4, lower bound: -95.4241643, upper bound: 95.4213516
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 4, lower bound: -95.4233232, upper bound: 95.4241638
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.15
Output dim: 4, lower bound: -95.4213450, upper bound: 95.4241638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207666
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4179370
time: 0.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4179338
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4182176
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179370
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179338
time: 0.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4200404
time: 0.57 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4207666
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4200781
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4206732
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4207666
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4179338, upper bound: 95.4207260
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4207260
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4200404, upper bound: 95.4179370
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4179338
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4182176
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4182176
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179370
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4182176, upper bound: 95.4179338
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4207666, upper bound: 95.4200404
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.99
Output dim: 4, lower bound: -95.4179370, upper bound: 95.4200404

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485283, upper bound: 95.3483756
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3485370, upper bound: 95.3483745
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483912
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3483892
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483892, upper bound: 95.3483745
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -95.3483745, upper bound: 95.3485370
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -16.1209259, 62.1793175, -16.1209259, 62.1793175, -78.3002396, 78.3002396
1: -25.9997044, 73.2714386, -25.9997044, 73.2714386, -99.2711411, 99.2711411
2: -22.6758308, 76.9590302, -22.6758308, 76.9590302, -99.6348495, 99.6348495
3: -42.5341644, 71.9460983, -42.5341644, 71.9460983, -114.4802628, 114.4802475
4: -33.1754951, 74.7584076, -33.1754951, 74.7584076, -107.9338989, 107.9338989

Time for backsubstitution: 1.71 seconds
Binary search (step 10): status=Status.UNKNOWN, low=0.3330078, high=0.3331706, mid=0.3331706, abs_max=107.93389892578125
rel_dist={4: [-95.43856405789779, 95.43856405789779]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.33300779265118763
execution time: 1132.24 seconds
