## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 807.3886655422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953)
1: (-373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194)
2: (-542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741)
3: (-209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672)
4: (-604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148)

## BASE Result
execution time: IAR + LP analysis = 1.44 + 1.89 = 3.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -809.0067386, upper bound: 809.0067386


# Binary Search by BASE starts (time budget: 1196.68 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.0067385931752]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=1011.34521484375
rel_dist={4: [-809.0065995903992, 809.0065995903992]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=1011.34521484375
rel_dist={4: [-809.0063734281499, 809.00637342815]}

## Binary search (step 3) starts
Candidate diff: 0.0208333


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0208333, mid=0.0208333, abs_max=1011.34521484375
rel_dist={4: [-809.0059584983813, 809.0059584983815]}

## Binary search (step 4) starts
Candidate diff: 0.0104167


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0104167, mid=0.0104167, abs_max=1011.34521484375
rel_dist={4: [-809.0054003420871, 809.0054003420869]}

## Binary search (step 5) starts
Candidate diff: 0.0052083


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0052083, mid=0.0052083, abs_max=1011.34521484375
rel_dist={4: [-809.005006272113, 809.005006272113]}

## Binary search (step 6) starts
Candidate diff: 0.0026042


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0026042, mid=0.0026042, abs_max=1011.34521484375
rel_dist={4: [-809.0047726399445, 809.0047726399443]}

## Binary search (step 7) starts
Candidate diff: 0.0013021


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0013021, mid=0.0013021, abs_max=1011.34521484375
rel_dist={4: [-809.0045709298558, 809.0045709298556]}

## Binary search (step 8) starts
Candidate diff: 0.0006510


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0006510, mid=0.0006510, abs_max=1011.34521484375
rel_dist={4: [-809.0044539569961, 809.0044539569963]}

## Binary search (step 9) starts
Candidate diff: 0.0003255


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0003255, mid=0.0003255, abs_max=1011.34521484375
rel_dist={4: [-809.0043787668834, 809.0043787668833]}

## Binary search (step 10) starts
Candidate diff: 0.0001628


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0001628, mid=0.0001628, abs_max=1011.34521484375
rel_dist={4: [-809.0043408960324, 809.0043408960323]}

## Binary search (step 11) starts
Candidate diff: 0.0000814


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000814, mid=0.0000814, abs_max=1011.34521484375
rel_dist={4: [-809.0043219310879, 809.0043219310878]}

## Binary search (step 12) starts
Candidate diff: 0.0000407


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000407, mid=0.0000407, abs_max=1011.34521484375
rel_dist={4: [-809.0043123987718, 809.0043123987716]}

## Binary search (step 13) starts
Candidate diff: 0.0000203


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000203, mid=0.0000203, abs_max=1011.34521484375
rel_dist={4: [-809.0043076328072, 809.004307632807]}

## Binary search (step 14) starts
Candidate diff: 0.0000102


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000102, mid=0.0000102, abs_max=1011.34521484375
rel_dist={4: [-809.0043052501968, 809.0043052501967]}

## Binary search (step 15) starts
Candidate diff: 0.0000051


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000051, mid=0.0000051, abs_max=1011.34521484375
rel_dist={4: [-809.0043044480162, 809.0043040595815]}

## Binary search (step 16) starts
Candidate diff: 0.0000025


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000025, mid=0.0000025, abs_max=1011.34521484375
rel_dist={4: [-809.0043036511679, 809.0043034673447]}

## Binary search (step 17) starts
Candidate diff: 0.0000013


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000013, mid=0.0000013, abs_max=1011.34521484375
rel_dist={4: [-809.0043032290516, 809.0043031699583]}

## Binary search (step 18) starts
Candidate diff: 0.0000006


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000006, mid=0.0000006, abs_max=1011.34521484375
rel_dist={4: [-809.004306129907, 809.0043400607263]}

## Binary Search Result
Binary search time: 62.06 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1134.62 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0066233, upper bound: 809.0064846
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0064846, upper bound: 809.0066233
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 4, lower bound: -809.0066233, upper bound: 809.0064846
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 4, lower bound: -809.0064846, upper bound: 809.0066233

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0037847, upper bound: 809.0036309
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0037855, upper bound: 809.0025058
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0060894, upper bound: 809.0060965
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0060908, upper bound: 809.0057312
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 4, lower bound: -809.0037847, upper bound: 809.0036309
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 4, lower bound: -809.0037855, upper bound: 809.0025058
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 4, lower bound: -809.0060894, upper bound: 809.0060965
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 4, lower bound: -809.0060908, upper bound: 809.0057312

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811570, upper bound: 808.5813259
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0028897, upper bound: 809.0023400
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0032552, upper bound: 809.0021488
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0021488, upper bound: 809.0032552
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0032488, upper bound: 809.0032587
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9974425, upper bound: 808.9980648
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9984189, upper bound: 808.9976010
time: 0.63 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -808.5811570, upper bound: 808.5813259
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -809.0028897, upper bound: 809.0023400
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -809.0032552, upper bound: 809.0021488
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -809.0021488, upper bound: 809.0032552
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -809.0032488, upper bound: 809.0032587
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -808.9974425, upper bound: 808.9980648
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 4, lower bound: -808.9984189, upper bound: 808.9976010

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.9832173, upper bound: 804.9832173
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.9832173, upper bound: 804.9832173
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3416749, upper bound: 808.3416882
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3410907, upper bound: 808.3416882
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9207666, upper bound: 806.9207677
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9207666, upper bound: 806.9208137
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9952580, upper bound: 808.9947622
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9958538, upper bound: 808.9947622
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0016717, upper bound: 809.0027782
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0016717, upper bound: 809.0019371
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0019071, upper bound: 809.0031422
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0031981, upper bound: 809.0032587
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9145254, upper bound: 806.9145031
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9145058, upper bound: 806.9145031
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2187065, upper bound: 808.2179758
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2187065, upper bound: 808.2179758
time: 0.62 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 4, lower bound: -804.9832173, upper bound: 804.9832173
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 4, lower bound: -804.9832173, upper bound: 804.9832173
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -808.3416749, upper bound: 808.3416882
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -808.3410907, upper bound: 808.3416882
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 4, lower bound: -806.9207666, upper bound: 806.9207677
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 4, lower bound: -806.9207666, upper bound: 806.9208137
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -808.9952580, upper bound: 808.9947622
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -808.9958538, upper bound: 808.9947622
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -809.0016717, upper bound: 809.0027782
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -809.0016717, upper bound: 809.0019371
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -809.0019071, upper bound: 809.0031422
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -809.0031981, upper bound: 809.0032587
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 4, lower bound: -806.9145254, upper bound: 806.9145031
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 4, lower bound: -806.9145058, upper bound: 806.9145031
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -808.2187065, upper bound: 808.2179758
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 4, lower bound: -808.2187065, upper bound: 808.2179758

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3322232, upper bound: 808.3321922
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3322115, upper bound: 808.3322070
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3404302, upper bound: 808.3413571
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3404302, upper bound: 808.3404478
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9782983, upper bound: 808.9781819
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9786269, upper bound: 808.9781819
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9958473, upper bound: 808.9945318
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9958242, upper bound: 808.9945318
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9187191, upper bound: 806.9187192
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9187191, upper bound: 806.9187677
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886239, upper bound: 808.9889315
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9886239, upper bound: 808.9888026
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9834078, upper bound: 808.9846066
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9834078, upper bound: 808.9834740
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9840866, upper bound: 808.9846449
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9846805, upper bound: 808.9846582
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019570, upper bound: 808.2013030
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019031, upper bound: 808.2013030
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756862, upper bound: 808.1756862
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1764169, upper bound: 808.1756862
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.3322232, upper bound: 808.3321922
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.3322115, upper bound: 808.3322070
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.3404302, upper bound: 808.3413571
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.3404302, upper bound: 808.3404478
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9782983, upper bound: 808.9781819
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9786269, upper bound: 808.9781819
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9958473, upper bound: 808.9945318
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9958242, upper bound: 808.9945318
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.16
Output dim: 4, lower bound: -806.9187191, upper bound: 806.9187192
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.16
Output dim: 4, lower bound: -806.9187191, upper bound: 806.9187677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9886239, upper bound: 808.9889315
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9886239, upper bound: 808.9888026
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9834078, upper bound: 808.9846066
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9834078, upper bound: 808.9834740
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9840866, upper bound: 808.9846449
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.9846805, upper bound: 808.9846582
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.2019570, upper bound: 808.2013030
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.2019031, upper bound: 808.2013030
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.1756862, upper bound: 808.1756862
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.16
Output dim: 4, lower bound: -808.1764169, upper bound: 808.1756862

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3250744, upper bound: 808.3255742
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3258341, upper bound: 808.3250744
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3308376, upper bound: 808.3318730
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3308376, upper bound: 808.3308376
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3308376, upper bound: 808.3318706
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3308376, upper bound: 808.3318730
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3343631, upper bound: 808.3343585
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3343444, upper bound: 808.3343797
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9777288, upper bound: 808.9777288
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9778452, upper bound: 808.9777288
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9771871, upper bound: 808.9766619
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9768714, upper bound: 808.9766619
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756026, upper bound: 808.1756026
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756026, upper bound: 808.1756026
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9788364, upper bound: 808.9777004
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9789513, upper bound: 808.9777004
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7604777, upper bound: 806.7604766
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7604766, upper bound: 806.7604766
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9777288, upper bound: 808.9779075
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9777288, upper bound: 808.9777288
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9207666, upper bound: 806.9207666
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9207666, upper bound: 806.9208146
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9541819, upper bound: 808.9542391
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9541819, upper bound: 808.9541819
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9831518, upper bound: 808.9841752
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9836166, upper bound: 808.9829379
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9831396, upper bound: 808.9841885
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9842107, upper bound: 808.9832196
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577449
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577449
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013111, upper bound: 808.2008705
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014852, upper bound: 808.2008705
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756026, upper bound: 808.1756026
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1756026, upper bound: 808.1756026
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577449
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577449, upper bound: 808.1577449
time: 0.64 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.19 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.3250744, upper bound: 808.3255742
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.3258341, upper bound: 808.3250744
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.3308376, upper bound: 808.3318730
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.3308376, upper bound: 808.3308376
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.3308376, upper bound: 808.3318706
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.3308376, upper bound: 808.3318730
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.3343631, upper bound: 808.3343585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.3343444, upper bound: 808.3343797
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9777288, upper bound: 808.9777288
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9778452, upper bound: 808.9777288
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9771871, upper bound: 808.9766619
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9768714, upper bound: 808.9766619
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1756026, upper bound: 808.1756026
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1756026, upper bound: 808.1756026
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9788364, upper bound: 808.9777004
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9789513, upper bound: 808.9777004
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -806.7604777, upper bound: 806.7604766
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -806.7604766, upper bound: 806.7604766
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9777288, upper bound: 808.9779075
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9777288, upper bound: 808.9777288
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -806.9207666, upper bound: 806.9207666
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.19
Output dim: 4, lower bound: -806.9207666, upper bound: 806.9208146
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9541819, upper bound: 808.9542391
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9541819, upper bound: 808.9541819
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9831518, upper bound: 808.9841752
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9836166, upper bound: 808.9829379
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9831396, upper bound: 808.9841885
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.9842107, upper bound: 808.9832196
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577449
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577449
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.2013111, upper bound: 808.2008705
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.2014852, upper bound: 808.2008705
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1756026, upper bound: 808.1756026
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1756026, upper bound: 808.1756026
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577449
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.19
Output dim: 4, lower bound: -808.1577449, upper bound: 808.1577449

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3253801
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3249016, upper bound: 808.3253291
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1579253, upper bound: 808.1577449
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583992, upper bound: 808.1577449
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1873934, upper bound: 808.1882927
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1873934, upper bound: 808.1873934
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247792, upper bound: 808.3247475
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3247475
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3255144
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3255196
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247294, upper bound: 808.3257928
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247294, upper bound: 808.3253846
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247294, upper bound: 808.3247502
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247294, upper bound: 808.3247294
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3246649, upper bound: 808.3246760
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3246649, upper bound: 808.3246759
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9762103, upper bound: 808.9762103
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9762103, upper bound: 808.9762103
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9763020, upper bound: 808.9762103
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9763520, upper bound: 808.9762103
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9459637, upper bound: 808.9454320
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9455292, upper bound: 808.9454320
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577920, upper bound: 808.1577449
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577449, upper bound: 808.1577449
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9123016, upper bound: 806.9123016
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9123016, upper bound: 806.9123016
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7484305, upper bound: 806.7484305
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7484305, upper bound: 806.7484305
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9762103, upper bound: 808.9762103
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9762103, upper bound: 808.9762103
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9537285, upper bound: 808.9537857
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9537285, upper bound: 808.9537496
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248888
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248889
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9187208, upper bound: 806.9187191
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9187208, upper bound: 806.9187669
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3278459, upper bound: 808.3278246
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3278459, upper bound: 808.3278246
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9773291, upper bound: 808.9784566
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9774324, upper bound: 808.9783222
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9773382, upper bound: 808.9774996
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9784888, upper bound: 808.9774644
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583433, upper bound: 808.1577449
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577449
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1581710, upper bound: 808.1577304
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577304
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2012209, upper bound: 808.2008705
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013111, upper bound: 808.2008705
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005650, upper bound: 808.2005610
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011757, upper bound: 808.2005610
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574483, upper bound: 808.1574425
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574425
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.56 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3253801
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3249016, upper bound: 808.3253291
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1579253, upper bound: 808.1577449
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1583992, upper bound: 808.1577449
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1873934, upper bound: 808.1882927
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1873934, upper bound: 808.1873934
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3247792, upper bound: 808.3247475
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3247475
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3255144
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3255196
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3247294, upper bound: 808.3257928
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3247294, upper bound: 808.3253846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3247294, upper bound: 808.3247502
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3247294, upper bound: 808.3247294
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3246649, upper bound: 808.3246760
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3246649, upper bound: 808.3246759
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9762103, upper bound: 808.9762103
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9762103, upper bound: 808.9762103
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9763020, upper bound: 808.9762103
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9763520, upper bound: 808.9762103
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9459637, upper bound: 808.9454320
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9455292, upper bound: 808.9454320
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1577920, upper bound: 808.1577449
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1577449, upper bound: 808.1577449
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 4, lower bound: -806.9123016, upper bound: 806.9123016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 4, lower bound: -806.9123016, upper bound: 806.9123016
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 4, lower bound: -806.7484305, upper bound: 806.7484305
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 4, lower bound: -806.7484305, upper bound: 806.7484305
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9762103, upper bound: 808.9762103
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9762103, upper bound: 808.9762103
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9537285, upper bound: 808.9537857
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9537285, upper bound: 808.9537496
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248888
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 4, lower bound: -806.9187208, upper bound: 806.9187191
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 4, lower bound: -806.9187208, upper bound: 806.9187669
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3278459, upper bound: 808.3278246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.3278459, upper bound: 808.3278246
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9773291, upper bound: 808.9784566
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9774324, upper bound: 808.9783222
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9773382, upper bound: 808.9774996
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.9784888, upper bound: 808.9774644
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1583433, upper bound: 808.1577449
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577449
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1581710, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.2012209, upper bound: 808.2008705
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.2013111, upper bound: 808.2008705
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.2005650, upper bound: 808.2005610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.2011757, upper bound: 808.2005610
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1755911, upper bound: 808.1755911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1574483, upper bound: 808.1574425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3250438
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245411
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1581710
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577596
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1579120, upper bound: 808.1577304
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574425
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1810253
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1802276
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574425
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3253235
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3249929
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1580964
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574848
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1810264
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800990
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800995
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245459
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3253120, upper bound: 808.3245459
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245777, upper bound: 808.3245349
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7484305, upper bound: 806.7484305
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7484305, upper bound: 806.7484305
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9451263, upper bound: 808.9449834
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9450807, upper bound: 808.9449834
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7483563, upper bound: 806.7483563
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7483563, upper bound: 806.7483573
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7483564, upper bound: 806.7483563
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7483563, upper bound: 806.7483563
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577771, upper bound: 808.1577304
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583867, upper bound: 808.1577304
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583785, upper bound: 808.1577304
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580319, upper bound: 808.1577304
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9449834
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9449834
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7570851, upper bound: 806.7570851
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7570851, upper bound: 806.7570851
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9449834
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9450045
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245523
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245523
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753001, upper bound: 808.1752811
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753001, upper bound: 808.1752811
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9102542, upper bound: 806.9102521
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9102542, upper bound: 806.9102951
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9102521, upper bound: 806.9102528
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9102521, upper bound: 806.9102858
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9102713, upper bound: 806.9102521
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.9102615, upper bound: 806.9102521
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1761064, upper bound: 808.1752811
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1761064, upper bound: 808.1752811
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583433, upper bound: 808.1577304
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577304
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574379, upper bound: 808.1574280
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578686, upper bound: 808.1574280
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583762, upper bound: 808.1577304
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577304
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009114, upper bound: 808.2005610
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005709, upper bound: 808.2005610
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010016, upper bound: 808.2005610
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005650, upper bound: 808.2005610
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574320, upper bound: 808.1574280
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1582526, upper bound: 808.1577304
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1579990, upper bound: 808.1577304
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574483, upper bound: 808.1574425
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.59 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3250438
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245411
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1581710
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577596
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1579120, upper bound: 808.1577304
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574425
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1810253
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1802276
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574425
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3253235
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3249929
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1580964
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574848
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1810264
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800990
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800995
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245459
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3253120, upper bound: 808.3245459
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245777, upper bound: 808.3245349
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.7484305, upper bound: 806.7484305
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.7484305, upper bound: 806.7484305
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.9451263, upper bound: 808.9449834
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.9450807, upper bound: 808.9449834
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.7483563, upper bound: 806.7483563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.7483563, upper bound: 806.7483573
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.7483564, upper bound: 806.7483563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.7483563, upper bound: 806.7483563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577771, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1583867, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1583785, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1580319, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9449834
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9449834
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.7570851, upper bound: 806.7570851
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.7570851, upper bound: 806.7570851
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9449834
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9450045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245523
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245523
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1753001, upper bound: 808.1752811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1753001, upper bound: 808.1752811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.9102542, upper bound: 806.9102521
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.9102542, upper bound: 806.9102951
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.9102521, upper bound: 806.9102528
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.9102521, upper bound: 806.9102858
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.9102713, upper bound: 806.9102521
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.06
Output dim: 4, lower bound: -806.9102615, upper bound: 806.9102521
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1761064, upper bound: 808.1752811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1761064, upper bound: 808.1752811
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1583433, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574379, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1578686, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1583762, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.2009114, upper bound: 808.2005610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.2005709, upper bound: 808.2005610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.2010016, upper bound: 808.2005610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.2005650, upper bound: 808.2005610
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574320, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1582526, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1579990, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574483, upper bound: 808.1574425
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.06
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.0067385931752]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0592859, upper bound: 806.0592859
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0592859, upper bound: 806.0592859
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.42 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.42
Output dim: 4, lower bound: -806.0592859, upper bound: 806.0592859
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.42
Output dim: 4, lower bound: -806.0592859, upper bound: 806.0592859
Binary search (step 1): status=Status.VERIFIED, low=0.0833333, high=0.1666667, mid=0.0833333, abs_max=1011.34521484375
rel_dist={4: [-809.0065995903992, 809.0065995903992]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9909785, upper bound: 808.9909785
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9909785, upper bound: 808.9910511
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.45 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.45
Output dim: 4, lower bound: -808.9909785, upper bound: 808.9909785
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.45
Output dim: 4, lower bound: -808.9909785, upper bound: 808.9910511

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8490056, upper bound: 806.8490055
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8490056, upper bound: 806.8490055
time: 0.85 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9600268, upper bound: 808.9600797
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9598769, upper bound: 808.9600529
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.88
Output dim: 4, lower bound: -806.8490056, upper bound: 806.8490055
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.88
Output dim: 4, lower bound: -806.8490056, upper bound: 806.8490055
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 4, lower bound: -808.9600268, upper bound: 808.9600797
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.88
Output dim: 4, lower bound: -808.9598769, upper bound: 808.9600529

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0590408, upper bound: 806.0592643
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0590408, upper bound: 806.0592643
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8391815, upper bound: 806.8392259
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8391788, upper bound: 806.8392259
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.81 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -806.0590408, upper bound: 806.0592643
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -806.0590408, upper bound: 806.0592643
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -806.8391815, upper bound: 806.8392259
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 4, lower bound: -806.8391788, upper bound: 806.8392259
Binary search (step 2): status=Status.VERIFIED, low=0.1250000, high=0.1666667, mid=0.1250000, abs_max=1011.34521484375
rel_dist={4: [-809.0067279617358, 809.0067279617358]}

## Binary search (step 3) starts
Candidate diff: 0.1458333


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8736456, upper bound: 806.8736456
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8736456, upper bound: 806.8736456
time: 0.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.63 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.63
Output dim: 4, lower bound: -806.8736456, upper bound: 806.8736456
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.63
Output dim: 4, lower bound: -806.8736456, upper bound: 806.8736456
Binary search (step 3): status=Status.VERIFIED, low=0.1458333, high=0.1666667, mid=0.1458333, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.0067385931752]}

## Binary search (step 4) starts
Candidate diff: 0.1562500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9989743, upper bound: 808.9990492
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9990492, upper bound: 808.9989743
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 4, lower bound: -808.9989743, upper bound: 808.9990492
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 4, lower bound: -808.9990492, upper bound: 808.9989743

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9808413, upper bound: 808.9809242
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9796858, upper bound: 808.9808916
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.3523970, upper bound: 806.3523925
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.3523970, upper bound: 806.3523925
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.54 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 4, lower bound: -808.9808413, upper bound: 808.9809242
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 4, lower bound: -808.9796858, upper bound: 808.9808916
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 2.54
Output dim: 4, lower bound: -806.3523970, upper bound: 806.3523925
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 2.54
Output dim: 4, lower bound: -806.3523970, upper bound: 806.3523925

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.4985770, upper bound: 804.4971967
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -804.4985770, upper bound: 804.4979423
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9794029, upper bound: 808.9806906
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9794006, upper bound: 808.9808257
time: 0.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 4, lower bound: -804.4985770, upper bound: 804.4971967
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 4, lower bound: -804.4985770, upper bound: 804.4979423
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 4, lower bound: -808.9794029, upper bound: 808.9806906
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 4, lower bound: -808.9794006, upper bound: 808.9808257

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9780602, upper bound: 808.9795009
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9780588, upper bound: 808.9795371
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9789204, upper bound: 808.9803536
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9789204, upper bound: 808.9797116
time: 0.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 4, lower bound: -808.9780602, upper bound: 808.9795009
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 4, lower bound: -808.9780588, upper bound: 808.9795371
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 4, lower bound: -808.9789204, upper bound: 808.9803536
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 4, lower bound: -808.9789204, upper bound: 808.9797116

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9774808, upper bound: 808.9790309
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9775902, upper bound: 808.9778905
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5265005, upper bound: 808.5254458
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5265005, upper bound: 808.5266155
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.3520546, upper bound: 806.3520634
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.3520546, upper bound: 806.3520634
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5252917, upper bound: 808.5250742
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5257790, upper bound: 808.5256406
time: 0.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -808.9774808, upper bound: 808.9790309
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -808.9775902, upper bound: 808.9778905
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -808.5265005, upper bound: 808.5254458
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -808.5265005, upper bound: 808.5266155
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 4, lower bound: -806.3520546, upper bound: 806.3520634
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.90
Output dim: 4, lower bound: -806.3520546, upper bound: 806.3520634
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -808.5252917, upper bound: 808.5250742
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.90
Output dim: 4, lower bound: -808.5257790, upper bound: 808.5256406

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5250664, upper bound: 808.5250664
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5250664, upper bound: 808.5261997
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9450600, upper bound: 808.9453692
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9451609, upper bound: 808.9453131
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2087664, upper bound: 808.2086065
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2086926, upper bound: 808.2086065
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2087664, upper bound: 808.2087909
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2086771, upper bound: 808.2086065
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7982113, upper bound: 806.7981749
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7981780, upper bound: 806.7981749
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2176444, upper bound: 808.2176323
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2176444, upper bound: 808.2176323
time: 0.72 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.5250664, upper bound: 808.5250664
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.5250664, upper bound: 808.5261997
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.9450600, upper bound: 808.9453692
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.9451609, upper bound: 808.9453131
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.2087664, upper bound: 808.2086065
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.2086926, upper bound: 808.2086065
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.2087664, upper bound: 808.2087909
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.2086771, upper bound: 808.2086065
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 4, lower bound: -806.7982113, upper bound: 806.7981749
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.84
Output dim: 4, lower bound: -806.7981780, upper bound: 806.7981749
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.2176444, upper bound: 808.2176323
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.84
Output dim: 4, lower bound: -808.2176444, upper bound: 808.2176323

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5177600, upper bound: 808.5177600
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5177600, upper bound: 808.5177600
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.3519101, upper bound: 806.3518960
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.3519101, upper bound: 806.3518960
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9450600, upper bound: 808.9453692
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9450600, upper bound: 808.9450600
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8243386, upper bound: 806.8243395
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8243354, upper bound: 806.8243403
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2008705, upper bound: 808.2008705
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009770, upper bound: 808.2008705
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1804961, upper bound: 808.1804101
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1804910, upper bound: 808.1804101
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2082565, upper bound: 808.2084409
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2084163, upper bound: 808.2082565
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2082565, upper bound: 808.2082565
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2083271, upper bound: 808.2082565
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753625, upper bound: 808.1753625
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1754089, upper bound: 808.1753625
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753625, upper bound: 808.1753625
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753747, upper bound: 808.1753625
time: 0.61 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.5177600, upper bound: 808.5177600
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.5177600, upper bound: 808.5177600
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.66
Output dim: 4, lower bound: -806.3519101, upper bound: 806.3518960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.66
Output dim: 4, lower bound: -806.3519101, upper bound: 806.3518960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.9450600, upper bound: 808.9453692
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.9450600, upper bound: 808.9450600
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.66
Output dim: 4, lower bound: -806.8243386, upper bound: 806.8243395
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.66
Output dim: 4, lower bound: -806.8243354, upper bound: 806.8243403
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.2008705, upper bound: 808.2008705
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.2009770, upper bound: 808.2008705
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.1804961, upper bound: 808.1804101
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.1804910, upper bound: 808.1804101
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.2082565, upper bound: 808.2084409
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.2084163, upper bound: 808.2082565
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.2082565, upper bound: 808.2082565
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.2083271, upper bound: 808.2082565
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.1753625, upper bound: 808.1753625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.1754089, upper bound: 808.1753625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.1753625, upper bound: 808.1753625
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.66
Output dim: 4, lower bound: -808.1753747, upper bound: 808.1753625

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.1469844, upper bound: 806.1469844
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.1469844, upper bound: 806.1469844
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9453246
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9450735, upper bound: 808.9451990
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8021180, upper bound: 806.8021180
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8021180, upper bound: 806.8021180
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578369, upper bound: 808.1577304
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577827, upper bound: 808.1577304
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1801652, upper bound: 808.1800792
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577771, upper bound: 808.1577304
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1802276
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1801498, upper bound: 808.1800792
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753430, upper bound: 808.1752811
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574934, upper bound: 808.1574280
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.74
Output dim: 4, lower bound: -806.1469844, upper bound: 806.1469844
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.74
Output dim: 4, lower bound: -806.1469844, upper bound: 806.1469844
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9453246
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.9450735, upper bound: 808.9451990
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.74
Output dim: 4, lower bound: -806.8021180, upper bound: 806.8021180
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.74
Output dim: 4, lower bound: -806.8021180, upper bound: 806.8021180
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1578369, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1577827, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1801652, upper bound: 808.1800792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1577771, upper bound: 808.1577304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1802276
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1801498, upper bound: 808.1800792
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1753430, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1574934, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.74
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1576096, upper bound: 808.1574280
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1575345, upper bound: 808.1574280
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1577524
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1575345, upper bound: 808.1574280
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1574280
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574753, upper bound: 808.1574280
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1575002, upper bound: 808.1574280
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574627, upper bound: 808.1574280
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574383, upper bound: 808.1574280
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.65 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.60
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.60
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1576096, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1575345, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1577524
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1575345, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574753, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1575002, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574627, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574383, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.60
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 1.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1570546, upper bound: 808.1570171
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1570171, upper bound: 808.1570546
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 1.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 1.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0330943, upper bound: 808.0330943
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0330943, upper bound: 808.0330943
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 1.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9514764, upper bound: 807.9514688
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9514688, upper bound: 807.9514764
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 1.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573815, upper bound: 808.1574280
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1575962, upper bound: 808.1573815
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 1.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0330943, upper bound: 808.0330943
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0334727, upper bound: 808.0330943
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 1.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1011536, upper bound: 808.1011091
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1011091, upper bound: 808.1011536
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 31

Time for candidate selection: 1.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835546, upper bound: 808.0835052
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835052, upper bound: 808.0835546
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 1.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1097789, upper bound: 808.1097258
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1097258, upper bound: 808.1097789
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 1.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1541971, upper bound: 808.1541523
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1541925, upper bound: 808.1541971
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 43

Time for candidate selection: 1.08 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1561752, upper bound: 808.1561847
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1561847, upper bound: 808.1561752
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 1.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0836603, upper bound: 808.0835052
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835052, upper bound: 808.0835546
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 9

Time for candidate selection: 1.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 1.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7315145, upper bound: 807.7314196
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7315145, upper bound: 807.7314196
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 1.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1570546, upper bound: 808.1570171
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1570171, upper bound: 808.1570546
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 1.09 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835546, upper bound: 808.0835052
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835052, upper bound: 808.0835546
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 1.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835546, upper bound: 808.0835052
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835052, upper bound: 808.0835546
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 17

Time for candidate selection: 1.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1011536, upper bound: 808.1011091
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1011091, upper bound: 808.1011536
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 1.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7314196, upper bound: 807.7314196
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7314196, upper bound: 807.7314196
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 18

Time for candidate selection: 1.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573685, upper bound: 808.1574186
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574186, upper bound: 808.1573685
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 1.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573815, upper bound: 808.1574280
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573815
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 1.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 1.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1097789, upper bound: 808.1098982
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1097258, upper bound: 808.1101354
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 1.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7314196, upper bound: 807.7314196
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.7314196, upper bound: 807.7314196
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 1.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 1.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 1.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 16

Time for candidate selection: 1.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9514764, upper bound: 807.9514688
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9514688, upper bound: 807.9514764
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 1.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9515865, upper bound: 807.9514688
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9514688, upper bound: 807.9514764
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 1.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835546, upper bound: 808.0835052
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835052, upper bound: 808.0835546
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 1.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835546, upper bound: 808.0835113
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0835052, upper bound: 808.0835546
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 1.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574024
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574024, upper bound: 808.1574280
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 1.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1097789, upper bound: 808.1097258
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1097258, upper bound: 808.1097789
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 1.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9514764, upper bound: 807.9514688
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -807.9514688, upper bound: 807.9514764
time: 2.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 1.13 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0330943, upper bound: 808.0330943
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0331623, upper bound: 808.0330943
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 1.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0200519, upper bound: 808.0199721
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0199721, upper bound: 808.0200110
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 25

Time for candidate selection: 1.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1097789, upper bound: 808.1097258
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1097258, upper bound: 808.1097789
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 1.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0200110, upper bound: 808.0199721
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0199721, upper bound: 808.0200110
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 1.14 seconds

### Candidate
type: RSZ, layer: 3, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1561752, upper bound: 808.1561847
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1561847, upper bound: 808.1561752
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 36

Time for candidate selection: 1.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573815, upper bound: 808.1574280
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573815
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 1.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573685, upper bound: 808.1574186
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574186, upper bound: 808.1573685
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 35
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 20

Time for candidate selection: 1.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0330943, upper bound: 808.0330943
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0330943, upper bound: 808.0330943
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 36
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 26
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 20
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 25
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 35

Time for candidate selection: 1.25 seconds

### Candidate
type: RSZ, layer: 3, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0330943, upper bound: 808.0330943
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.0330943, upper bound: 808.0330943
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds
Binary search (step 4): status=Status.UNKNOWN, low=0.1458333, high=0.1562500, mid=0.1562500, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.006738593175]}

## Binary search (step 5) starts
Candidate diff: 0.1510417


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8736456, upper bound: 806.8736456
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8736456, upper bound: 806.8736456
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.67 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.67
Output dim: 4, lower bound: -806.8736456, upper bound: 806.8736456
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.67
Output dim: 4, lower bound: -806.8736456, upper bound: 806.8736456
Binary search (step 5): status=Status.VERIFIED, low=0.1510417, high=0.1562500, mid=0.1510417, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.0067385931752]}

## Binary search (step 6) starts
Candidate diff: 0.1536458


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0593958, upper bound: 806.0593958
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0593958, upper bound: 806.0593958
time: 0.65 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.33 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.33
Output dim: 4, lower bound: -806.0593958, upper bound: 806.0593958
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.33
Output dim: 4, lower bound: -806.0593958, upper bound: 806.0593958
Binary search (step 6): status=Status.VERIFIED, low=0.1536458, high=0.1562500, mid=0.1536458, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.006738593175]}

## Binary search (step 7) starts
Candidate diff: 0.1549479


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9881757, upper bound: 808.9881807
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9881757, upper bound: 808.9881757
time: 0.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -808.9881757, upper bound: 808.9881807
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -808.9881757, upper bound: 808.9881757

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9600529, upper bound: 808.9598769
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9600268, upper bound: 808.9600797
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9600797, upper bound: 808.9600268
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9598769, upper bound: 808.9600529
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 4, lower bound: -808.9600529, upper bound: 808.9598769
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 4, lower bound: -808.9600268, upper bound: 808.9600797
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 4, lower bound: -808.9600797, upper bound: 808.9600268
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.78
Output dim: 4, lower bound: -808.9598769, upper bound: 808.9600529

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9590134, upper bound: 808.9595333
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9596007, upper bound: 808.9584175
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2005495, upper bound: 806.2005562
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.2005495, upper bound: 806.2005562
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9597909, upper bound: 808.9600268
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9600797, upper bound: 808.9598777
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8391815, upper bound: 806.8392259
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8391788, upper bound: 806.8392259
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.71 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 4, lower bound: -808.9590134, upper bound: 808.9595333
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 4, lower bound: -808.9596007, upper bound: 808.9584175
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.71
Output dim: 4, lower bound: -806.2005495, upper bound: 806.2005562
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.71
Output dim: 4, lower bound: -806.2005495, upper bound: 806.2005562
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 4, lower bound: -808.9597909, upper bound: 808.9600268
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 4, lower bound: -808.9600797, upper bound: 808.9598777
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 2.71
Output dim: 4, lower bound: -806.8391815, upper bound: 806.8392259
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 2.71
Output dim: 4, lower bound: -806.8391788, upper bound: 806.8392259

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9582281, upper bound: 808.9590851
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9582281, upper bound: 808.9579693
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.9052514, upper bound: 805.9053826
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -805.9052514, upper bound: 805.9053826
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8167356, upper bound: 806.8167802
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8167333, upper bound: 806.8167802
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9598559, upper bound: 808.9595144
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9597180, upper bound: 808.9584175
time: 0.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 4, lower bound: -808.9582281, upper bound: 808.9590851
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 4, lower bound: -808.9582281, upper bound: 808.9579693
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 4, lower bound: -805.9052514, upper bound: 805.9053826
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 4, lower bound: -805.9052514, upper bound: 805.9053826
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 4, lower bound: -806.8167356, upper bound: 806.8167802
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 4, lower bound: -806.8167333, upper bound: 806.8167802
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 4, lower bound: -808.9598559, upper bound: 808.9595144
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.77
Output dim: 4, lower bound: -808.9597180, upper bound: 808.9584175

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3605363, upper bound: 808.3611153
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3605363, upper bound: 808.3611224
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9458215, upper bound: 808.9455716
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9461687, upper bound: 808.9455716
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9544385, upper bound: 808.9544944
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9558502, upper bound: 808.9555129
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9461155, upper bound: 808.9460207
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9472409, upper bound: 808.9460207
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 4, lower bound: -808.3605363, upper bound: 808.3611153
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 4, lower bound: -808.3605363, upper bound: 808.3611224
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 4, lower bound: -808.9458215, upper bound: 808.9455716
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 4, lower bound: -808.9461687, upper bound: 808.9455716
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 4, lower bound: -808.9544385, upper bound: 808.9544944
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 4, lower bound: -808.9558502, upper bound: 808.9555129
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 4, lower bound: -808.9461155, upper bound: 808.9460207
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.61
Output dim: 4, lower bound: -808.9472409, upper bound: 808.9460207

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3553726, upper bound: 808.3554136
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3553837, upper bound: 808.3559627
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009272, upper bound: 808.2013928
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009195, upper bound: 808.2010366
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9452318
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9454873, upper bound: 808.9452318
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465384
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465385
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9541819, upper bound: 808.9542317
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9541819, upper bound: 808.9542391
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3263000, upper bound: 808.3251528
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3263000, upper bound: 808.3251528
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9457710, upper bound: 808.9456808
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9457787, upper bound: 808.9456808
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1586231, upper bound: 808.1580889
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1585921, upper bound: 808.1580889
time: 0.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.3553726, upper bound: 808.3554136
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.3553837, upper bound: 808.3559627
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.2009272, upper bound: 808.2013928
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.2009195, upper bound: 808.2010366
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9452318
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.9454873, upper bound: 808.9452318
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465384
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.39
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465385
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.9541819, upper bound: 808.9542317
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.9541819, upper bound: 808.9542391
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.3263000, upper bound: 808.3251528
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.3263000, upper bound: 808.3251528
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.9457710, upper bound: 808.9456808
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.9457787, upper bound: 808.9456808
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.1586231, upper bound: 808.1580889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.39
Output dim: 4, lower bound: -808.1585921, upper bound: 808.1580889

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248165
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248575
time: 0.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009272, upper bound: 808.2013928
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009195, upper bound: 808.2010366
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009195, upper bound: 808.2012830
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009272, upper bound: 808.2013928
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578204
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1579037
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9452318
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9452318
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9452342, upper bound: 808.9449834
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9452000, upper bound: 808.9449834
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248921
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248937
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248888
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248889
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3254229, upper bound: 808.3248165
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3259638, upper bound: 808.3248165
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3254148, upper bound: 808.3248165
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3259638, upper bound: 808.3248165
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580889, upper bound: 808.1580889
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580889, upper bound: 808.1580889
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1581354, upper bound: 808.1580889
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1581354, upper bound: 808.1580889
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1585621, upper bound: 808.1580889
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1586231, upper bound: 808.1580889
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577903, upper bound: 808.1577865
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1582898, upper bound: 808.1577865
time: 0.69 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248165
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248575
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.2009272, upper bound: 808.2013928
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.2009195, upper bound: 808.2010366
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.2009195, upper bound: 808.2012830
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.2009272, upper bound: 808.2013928
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578204
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1579037
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9452318
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9452318
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.9452342, upper bound: 808.9449834
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.9452000, upper bound: 808.9449834
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248921
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248937
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248888
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3248711, upper bound: 808.3248889
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3254229, upper bound: 808.3248165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3259638, upper bound: 808.3248165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3254148, upper bound: 808.3248165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.3259638, upper bound: 808.3248165
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1580889, upper bound: 808.1580889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1580889, upper bound: 808.1580889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1581354, upper bound: 808.1580889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1581354, upper bound: 808.1580889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1585621, upper bound: 808.1580889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1586231, upper bound: 808.1580889
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1577903, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.75
Output dim: 4, lower bound: -808.1582898, upper bound: 808.1577865

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578035
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578161
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577943, upper bound: 808.1582598
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1579037
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1581500
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578233
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1582466
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577943, upper bound: 808.1582598
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578204
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578094
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578222
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1579037
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465384
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465385
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005719, upper bound: 808.2005610
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005719, upper bound: 808.2005610
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577505
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577520
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577408
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577408
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577964, upper bound: 808.1577865
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1582787, upper bound: 808.1577865
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578191, upper bound: 808.1577865
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1585994, upper bound: 808.1577865
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577964, upper bound: 808.1577865
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1582787, upper bound: 808.1577865
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3249620, upper bound: 808.3245349
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3253285, upper bound: 808.3245349
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577955, upper bound: 808.1577865
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578331, upper bound: 808.1577865
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577957, upper bound: 808.1577304
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1582598, upper bound: 808.1577865
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583974, upper bound: 808.1577304
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577304
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574338, upper bound: 808.1574280
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577282, upper bound: 808.1574280
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
time: 0.69 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578035
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578161
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577943, upper bound: 808.1582598
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1579037
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1581500
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578233
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1582466
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577943, upper bound: 808.1582598
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578204
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578094
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1578222
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1579037
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.01
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465384
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.01
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465385
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.01
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.01
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.2005719, upper bound: 808.2005610
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.2005719, upper bound: 808.2005610
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577505
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577520
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577408
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577408
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577964, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1582787, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1578191, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1585994, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577964, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1582787, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.3249620, upper bound: 808.3245349
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.3253285, upper bound: 808.3245349
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577955, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1578331, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577957, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1582598, upper bound: 808.1577865
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1583974, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1583989, upper bound: 808.1577304
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1574338, upper bound: 808.1574280
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1577282, upper bound: 808.1574280
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.01
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574571
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574627
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574690
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1576967
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1580959
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574705
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1576096
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1579203
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574854
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574789
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574794
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580427
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580761
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1579873
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1580959
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574711
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574748
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574477
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574635
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574710
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574778
time: 1.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574709
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1576096
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds
Binary search (step 7): status=Status.UNKNOWN, low=0.1536458, high=0.1549479, mid=0.1549479, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.006738593175]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.15364582417532802
execution time: 1135.52 seconds
