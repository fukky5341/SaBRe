## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.088187946


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102)
1: (-0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898)
2: (-0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237)
3: (-0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035)
4: (-0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858)

## BASE Result
execution time: IAR + LP analysis = 1.70 + 0.90 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877


# Binary Search by BASE starts (time budget: 1197.41 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=0.10251016169786453
rel_dist={0: [-0.0899835175585181, 0.08998351755851813]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=0.10251016169786453
rel_dist={0: [-0.08996129100877424, 0.08996129100877422]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=0.10251016169786453
rel_dist={0: [-0.0899476251703134, 0.08994762517031346]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=0.10251016169786453
rel_dist={0: [-0.08994001998180878, 0.08994001998180878]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0031250, mid=0.0031250, abs_max=0.10251016169786453
rel_dist={0: [-0.08993124712088736, 0.08993124712088738]}

## Binary search (step 6) starts
Candidate diff: 0.0015625


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0015625, mid=0.0015625, abs_max=0.10251016169786453
rel_dist={0: [-0.08992489701688348, 0.08992489701674958]}

## Binary search (step 7) starts
Candidate diff: 0.0007812


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007812, mid=0.0007812, abs_max=0.10251016169786453
rel_dist={0: [-0.08991947978566078, 0.08991947978566078]}

## Binary search (step 8) starts
Candidate diff: 0.0003906


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003906, mid=0.0003906, abs_max=0.10251016169786453
rel_dist={0: [-0.0899161563368783, 0.08991615633673694]}

## Binary search (step 9) starts
Candidate diff: 0.0001953


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001953, mid=0.0001953, abs_max=0.10251016169786453
rel_dist={0: [-0.0899141627247981, 0.08991416272479807]}

## Binary search (step 10) starts
Candidate diff: 0.0000977


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000977, mid=0.0000977, abs_max=0.10251016169786453
rel_dist={0: [-0.08991292356463379, 0.08991292356463376]}

## Binary search (step 11) starts
Candidate diff: 0.0000488


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000488, mid=0.0000488, abs_max=0.10251016169786453
rel_dist={0: [-0.08991230398547434, 0.08991230398547431]}

## Binary search (step 12) starts
Candidate diff: 0.0000244


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000244, mid=0.0000244, abs_max=0.10251016169786453
rel_dist={0: [-0.08991199419703344, 0.08991199419702026]}

## Binary search (step 13) starts
Candidate diff: 0.0000122


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000122, mid=0.0000122, abs_max=0.10251016169786453
rel_dist={0: [-0.08991183930756332, 0.08991183930756333]}

## Binary search (step 14) starts
Candidate diff: 0.0000061


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000061, mid=0.0000061, abs_max=0.10251016169786453
rel_dist={0: [-0.08991176188726252, 0.08991176188726249]}

## Binary search (step 15) starts
Candidate diff: 0.0000031


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000031, mid=0.0000031, abs_max=0.10251016169786453
rel_dist={0: [-0.08991172476862014, 0.08991172321457086]}

## Binary search (step 16) starts
Candidate diff: 0.0000015


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000015, mid=0.0000015, abs_max=0.10251016169786453
rel_dist={0: [-0.08991170448511418, 0.08991173620782567]}

## Binary search (step 17) starts
Candidate diff: 0.0000008


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000008, mid=0.0000008, abs_max=0.10251016169786453
rel_dist={0: [-0.08991169739850569, 0.08991171335425518]}

## Binary Search Result
Binary search time: 43.75 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1153.65 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0893632
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0895442
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0893508
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0895022
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0893632
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0895442
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0893508
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0895022

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895371
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895198
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0893406
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0893406
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0894892
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0894892
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895371
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895198
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0893406
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0893406
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0894892
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0894892

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880122
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880122
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880076, upper bound: 0.0880213
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880099, upper bound: 0.0880213
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880203
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880203
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894787
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880102
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880102
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.32 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0880076, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0880099, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880203
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880102
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.32
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880102

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899440
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899253
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887967
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887967
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899440
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899253
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887967
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1.98
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887967

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0899161
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898879, upper bound: 0.0898872
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880203
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880102
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0899161
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0898879, upper bound: 0.0898872
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880203
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880102

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0893039
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880096
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0893039
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880096
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0893632
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0895442
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0893632
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0893508, upper bound: 0.0895442
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880317
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880317
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887871
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893284
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894910
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880317
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880317
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887871
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893284
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894910

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880213
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0893235
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0880044
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.05 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0893235
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0880044

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.01 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.01
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.01
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.01
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.01
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.01
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.01
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.01
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.01
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0894300
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0894300
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0895781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880317
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880317
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895285
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895204
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.27 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880317
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880317
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895285
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895204

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894971
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894787
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.25 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894971
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.26 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2000000, mid=0.1875000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 4) starts
Candidate diff: 0.1937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0894233
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899440
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899440, upper bound: 0.0899159
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.22 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0894233
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899440
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0899440, upper bound: 0.0899159

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895285
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895204
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895285, upper bound: 0.0894015
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894040, upper bound: 0.0895204
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895285
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895204
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0895285, upper bound: 0.0894015
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0894040, upper bound: 0.0895204

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894971
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894971
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880203
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
Binary search (step 4): status=Status.VERIFIED, low=0.1937500, high=0.2000000, mid=0.1937500, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 5) starts
Candidate diff: 0.1968750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887971
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887985
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887971
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887985
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887873
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887871
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887828
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887871
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.23
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887828

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880216
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880122
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880213
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880316
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880213
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 5): status=Status.VERIFIED, low=0.1968750, high=0.2000000, mid=0.1968750, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 6) starts
Candidate diff: 0.1984375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0894300
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0894300
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0895781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.21 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.21
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.21
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895285
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895204
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895285
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895204
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.00 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895285
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895204
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895285
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895204

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894971
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894787
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.02 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.02
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.02
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894971
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.02
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894787

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.08 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.08
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 6): status=Status.VERIFIED, low=0.1984375, high=0.2000000, mid=0.1984375, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 7) starts
Candidate diff: 0.1992187


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887913
time: 0.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887913
time: 0.26 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.20 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887913
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887913
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880317
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887913
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887847
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898879, upper bound: 0.0898872
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898873, upper bound: 0.0898872
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899161, upper bound: 0.0898873
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880108, upper bound: 0.0880320
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880317
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887913
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887847
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0898879, upper bound: 0.0898872
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0898873, upper bound: 0.0898872
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0899161, upper bound: 0.0898873
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880317
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894791, upper bound: 0.0893229
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.26 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880317
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0894791, upper bound: 0.0893229
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880203
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 7): status=Status.VERIFIED, low=0.1992187, high=0.2000000, mid=0.1992187, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 8) starts
Candidate diff: 0.1996094


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887971
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887985
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.09 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.09
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.09
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887971
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887985

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887873
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887871
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887843
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.03 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887873
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887871
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.03
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887843

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880316
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887843
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.26 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880316
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887843
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.30 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 8): status=Status.VERIFIED, low=0.1996094, high=0.2000000, mid=0.1996094, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 9) starts
Candidate diff: 0.1998047


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899482
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899423
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899482
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899423
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887873
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887873
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893429
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895198
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893508
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894987
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893284
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894910
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887873
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893429
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895198
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893508
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894987
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893284
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.08
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894910

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880122
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880213
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880213
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880103
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880103
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880105
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880198
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894787
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880103
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880105
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880198
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.07
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.34
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 9): status=Status.VERIFIED, low=0.1998047, high=0.2000000, mid=0.1998047, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 10) starts
Candidate diff: 0.1999023


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0894300
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0894300
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0895781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894064, upper bound: 0.0895718
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894064, upper bound: 0.0895604
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.24 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.0880317, upper bound: 0.0880320
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.0894064, upper bound: 0.0895718
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -0.0894064, upper bound: 0.0895604

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895371
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894987
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895204
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895204
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.27 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895371
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894987
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0894015, upper bound: 0.0895204
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895204

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880122
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880122
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.29 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880122
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880122
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880044
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.06 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.06
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880044
Binary search (step 10): status=Status.VERIFIED, low=0.1999023, high=0.2000000, mid=0.1999023, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 11) starts
Candidate diff: 0.1999512


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887985

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887967
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887967
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887967
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.04
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887967

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887913
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887871
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.05
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.05
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.05
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.05
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.05
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887913
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.05
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887871
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.05
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.05
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880317
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887843
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880317
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887843
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880044
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.37 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880044
Binary search (step 11): status=Status.VERIFIED, low=0.1999512, high=0.2000000, mid=0.1999512, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 12) starts
Candidate diff: 0.1999756


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0894300
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.61 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0894300
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0894300, upper bound: 0.0895781

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894064, upper bound: 0.0894292
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894064, upper bound: 0.0894064
time: 0.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0894064, upper bound: 0.0894292
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0894064, upper bound: 0.0894064
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880216
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880216
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880320
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880320
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880216
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880320
Binary search (step 12): status=Status.VERIFIED, low=0.1999756, high=0.2000000, mid=0.1999756, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 13) starts
Candidate diff: 0.1999878


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
time: 0.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0894233
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0894233
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0894233
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0894233
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 0, lower bound: -0.0894233, upper bound: 0.0895781

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0895442
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0894892
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880203
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880203
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895285
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895204
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880320
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0895442
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0893406, upper bound: 0.0894892
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0880102, upper bound: 0.0880203
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895285
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.06
Output dim: 0, lower bound: -0.0894025, upper bound: 0.0895204

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894971
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894971
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894787
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894971
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0894787
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894971
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0894787
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.05
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.35
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
Binary search (step 13): status=Status.VERIFIED, low=0.1999878, high=0.2000000, mid=0.1999878, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 14) starts
Candidate diff: 0.1999939


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899763
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899693
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899763
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899693

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899440
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899440
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899253
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899159
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899440
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899440
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899253
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899159

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895285, upper bound: 0.0894015
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894040, upper bound: 0.0895204
time: 0.32 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.33 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0895285, upper bound: 0.0894015
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0894040, upper bound: 0.0895204

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894787, upper bound: 0.0893411
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0894787, upper bound: 0.0893411
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.06
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880203
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.09 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.09
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
Binary search (step 14): status=Status.VERIFIED, low=0.1999939, high=0.2000000, mid=0.1999939, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 15) starts
Candidate diff: 0.1999969


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899763
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899693
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899763
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.64
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899693

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899440
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899440
time: 0.31 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899253
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899159
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899440
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899440
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 0, lower bound: -0.0899159, upper bound: 0.0899253
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.06
Output dim: 0, lower bound: -0.0899253, upper bound: 0.0899159

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0899161
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898879, upper bound: 0.0898872
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0899161
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0898879, upper bound: 0.0898872
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0887967, upper bound: 0.0887985
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887418

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887843
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887843
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.25
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 15): status=Status.VERIFIED, low=0.1999969, high=0.2000000, mid=0.1999969, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 16) starts
Candidate diff: 0.1999985


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899763
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899693
time: 0.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899763
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899693

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887971
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887971
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899423
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887971
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887971
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899423
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893429
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895198
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873
time: 0.31 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0887418, upper bound: 0.0887971
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0887985, upper bound: 0.0887967
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893429
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0895198
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887843
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894791
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0893039, upper bound: 0.0894789
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887847, upper bound: 0.0887828
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0887853, upper bound: 0.0887843
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -0.0893475, upper bound: 0.0894787

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880096
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880044
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880096
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880044
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.18
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
Binary search (step 16): status=Status.VERIFIED, low=0.1999985, high=0.2000000, mid=0.1999985, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 17) starts
Candidate diff: 0.1999992


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899763
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899693
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.65 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899763
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -0.0899693, upper bound: 0.0899693

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899482
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
time: 0.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887985
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887985
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899482
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887985
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0887971, upper bound: 0.0887985

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887873
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887873
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893508
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894987
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887828
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0880316
time: 0.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887873
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887873
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0893508
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -0.0893284, upper bound: 0.0894987
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887828
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.29
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880320
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.29
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0880316

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880216
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880122
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880122
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0893039
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0893229
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880105
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880122
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880122
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0893411, upper bound: 0.0893039
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0893235, upper bound: 0.0893229
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0880105
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.05 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.05
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.05
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.05
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.05
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.05
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.05
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.05
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.05
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
Binary search (step 17): status=Status.VERIFIED, low=0.1999992, high=0.2000000, mid=0.1999992, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1999992251396634
execution time: 701.48 seconds
