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
execution time: IAR + LP analysis = 1.65 + 1.91 = 3.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -809.0067386, upper bound: 809.0067386


# Binary Search by BASE starts (time budget: 1196.44 seconds, max iter: 100)

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
Binary search time: 66.81 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1129.63 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0038209, upper bound: 809.0039007
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0039007, upper bound: 809.0038209
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.35 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 4, lower bound: -809.0038209, upper bound: 809.0039007
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 4, lower bound: -809.0039007, upper bound: 809.0038209

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0037847, upper bound: 809.0036309
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0025058, upper bound: 809.0037855
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0037855, upper bound: 809.0025058
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0036309, upper bound: 809.0037847
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 4, lower bound: -809.0037847, upper bound: 809.0036309
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 4, lower bound: -809.0025058, upper bound: 809.0037855
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 4, lower bound: -809.0037855, upper bound: 809.0025058
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 4, lower bound: -809.0036309, upper bound: 809.0037847

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811570, upper bound: 808.5813259
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811632
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813270
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813270, upper bound: 808.5810333
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5810333
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813259, upper bound: 808.5811570
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5812327
time: 0.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.01 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -808.5811570, upper bound: 808.5813259
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811632
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813270
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -808.5813270, upper bound: 808.5810333
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5810333
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -808.5813259, upper bound: 808.5811570
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.01
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5812327

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197399, upper bound: 808.5197665
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197823, upper bound: 808.5193886
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197357, upper bound: 808.5198311
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197374, upper bound: 808.5193886
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5197434
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5198400
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5198400, upper bound: 808.5188205
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197434, upper bound: 808.5188205
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197374
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5198311, upper bound: 808.5197357
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197823
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197665, upper bound: 808.5197399
time: 0.74 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5197399, upper bound: 808.5197665
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5197823, upper bound: 808.5193886
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5197357, upper bound: 808.5198311
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5197374, upper bound: 808.5193886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5197434
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5198400
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5198400, upper bound: 808.5188205
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5197434, upper bound: 808.5188205
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197374
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5198311, upper bound: 808.5197357
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197823
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 4, lower bound: -808.5197665, upper bound: 808.5197399

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016339, upper bound: 808.2014357
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016339, upper bound: 808.2023070
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023309, upper bound: 808.2014515
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023309, upper bound: 808.2015041
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016339, upper bound: 808.2023036
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016339, upper bound: 808.2023077
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023078, upper bound: 808.2014811
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017733, upper bound: 808.2015041
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015116, upper bound: 808.2014357
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014811, upper bound: 808.2021921
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023078, upper bound: 808.2014357
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022444, upper bound: 808.2014817
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015116, upper bound: 808.2023145
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014810, upper bound: 808.2023145
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017733, upper bound: 808.2014851
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017733, upper bound: 808.2014908
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014908, upper bound: 808.2017733
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014851, upper bound: 808.2023078
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023145, upper bound: 808.2014810
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023145, upper bound: 808.2015116
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014817, upper bound: 808.2022444
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014357, upper bound: 808.2023078
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021921, upper bound: 808.2014811
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014357, upper bound: 808.2015116
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015041, upper bound: 808.2023309
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014811, upper bound: 808.2023309
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023077, upper bound: 808.2016339
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023036, upper bound: 808.2016339
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015041, upper bound: 808.2023309
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014515, upper bound: 808.2023309
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023070, upper bound: 808.2016339
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014357, upper bound: 808.2016339
time: 0.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2016339, upper bound: 808.2014357
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2016339, upper bound: 808.2023070
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023309, upper bound: 808.2014515
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023309, upper bound: 808.2015041
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2016339, upper bound: 808.2023036
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2016339, upper bound: 808.2023077
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023078, upper bound: 808.2014811
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2017733, upper bound: 808.2015041
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2015116, upper bound: 808.2014357
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014811, upper bound: 808.2021921
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023078, upper bound: 808.2014357
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2022444, upper bound: 808.2014817
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2015116, upper bound: 808.2023145
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014810, upper bound: 808.2023145
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2017733, upper bound: 808.2014851
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2017733, upper bound: 808.2014908
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014908, upper bound: 808.2017733
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014851, upper bound: 808.2023078
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023145, upper bound: 808.2014810
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023145, upper bound: 808.2015116
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014817, upper bound: 808.2022444
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014357, upper bound: 808.2023078
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2021921, upper bound: 808.2014811
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014357, upper bound: 808.2015116
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2015041, upper bound: 808.2023309
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014811, upper bound: 808.2023309
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023077, upper bound: 808.2016339
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023036, upper bound: 808.2016339
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2015041, upper bound: 808.2023309
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014515, upper bound: 808.2023309
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2023070, upper bound: 808.2016339
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.32
Output dim: 4, lower bound: -808.2014357, upper bound: 808.2016339

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1950232, upper bound: 808.1948166
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957243
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1950232, upper bound: 808.1948166
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957422, upper bound: 808.1948330
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948900
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957422, upper bound: 808.1948359
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957324
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1950232, upper bound: 808.1957008
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957330
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1950232, upper bound: 808.1957008
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948653
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957250, upper bound: 808.1948491
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948900
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951503, upper bound: 808.1948491
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948396, upper bound: 808.1948166
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948981, upper bound: 808.1948166
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948396, upper bound: 808.1955162
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948652, upper bound: 808.1948166
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957257, upper bound: 808.1948166
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948404
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1956689, upper bound: 808.1948166
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948378, upper bound: 808.1957387
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948981, upper bound: 808.1948166
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948378, upper bound: 808.1957387
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948645, upper bound: 808.1948166
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948696
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957250, upper bound: 808.1948166
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948756
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951503, upper bound: 808.1948166
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1951503
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948756, upper bound: 808.1948166
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957250
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948696, upper bound: 808.1948166
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948645
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957387, upper bound: 808.1948378
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948981
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957387, upper bound: 808.1948378
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1956689
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948404, upper bound: 808.1948166
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957257
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948652
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1955162, upper bound: 808.1948396
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948981
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948396
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948491, upper bound: 808.1957422
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948900, upper bound: 808.1948166
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948491, upper bound: 808.1957422
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948653, upper bound: 808.1948166
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957008, upper bound: 808.1950232
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957330, upper bound: 808.1948166
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957008, upper bound: 808.1950232
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948359, upper bound: 808.1957422
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948900, upper bound: 808.1948166
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948330, upper bound: 808.1957422
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1950232
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1957243, upper bound: 808.1948166
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1950232
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
time: 0.63 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1950232, upper bound: 808.1948166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957243
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1950232, upper bound: 808.1948166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957422, upper bound: 808.1948330
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948900
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957422, upper bound: 808.1948359
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957324
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1950232, upper bound: 808.1957008
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957330
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1950232, upper bound: 808.1957008
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957250, upper bound: 808.1948491
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948900
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1951503, upper bound: 808.1948491
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948396, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948981, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948396, upper bound: 808.1955162
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948652, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957257, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948404
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1956689, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948378, upper bound: 808.1957387
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948981, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948378, upper bound: 808.1957387
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948645, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948696
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957250, upper bound: 808.1948166
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948756
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1951503, upper bound: 808.1948166
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1951503
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948756, upper bound: 808.1948166
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957250
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948696, upper bound: 808.1948166
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948645
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957387, upper bound: 808.1948378
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948981
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957387, upper bound: 808.1948378
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1956689
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948404, upper bound: 808.1948166
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1957257
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948652
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1955162, upper bound: 808.1948396
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948981
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948396
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948491, upper bound: 808.1957422
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948900, upper bound: 808.1948166
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948491, upper bound: 808.1957422
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948653, upper bound: 808.1948166
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957008, upper bound: 808.1950232
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957330, upper bound: 808.1948166
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957008, upper bound: 808.1950232
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948359, upper bound: 808.1957422
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948900, upper bound: 808.1948166
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948330, upper bound: 808.1957422
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1950232
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1957243, upper bound: 808.1948166
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1950232
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.08
Output dim: 4, lower bound: -808.1948166, upper bound: 808.1948166

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1954108
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945117
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945195
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954287, upper bound: 808.1945132
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945765
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945129
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945223
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954287, upper bound: 808.1945132
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1813301
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1813562
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1953873
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945144
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1813592
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1813573
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1806688, upper bound: 808.1813230
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1804807, upper bound: 808.1804101
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577734
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577596
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583992, upper bound: 808.1577304
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577957
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577723
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577596
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1583982, upper bound: 808.1577413
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945106, upper bound: 808.1945031
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945261, upper bound: 808.1945031
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945095, upper bound: 808.1945031
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945846, upper bound: 808.1945031
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945106, upper bound: 808.1952006
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945261, upper bound: 808.1945031
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945088, upper bound: 808.1945031
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945088, upper bound: 808.1945031
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954122, upper bound: 808.1945031
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945263
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945106, upper bound: 808.1954252
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945243, upper bound: 808.1945031
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945095, upper bound: 808.1945031
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945095, upper bound: 808.1945031
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945106, upper bound: 808.1954252
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945243, upper bound: 808.1945031
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945088, upper bound: 808.1945031
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945510, upper bound: 808.1945031
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945560
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954115, upper bound: 808.1945031
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945621
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1948368
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1954115
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945510
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945088
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945243
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954252, upper bound: 808.1945106
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945846
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945095
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945243
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945106
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1953554
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945263, upper bound: 808.1945031
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1954122
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945517
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945088
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945261
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1952006, upper bound: 808.1945106
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945846
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945095
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945261
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945106
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577413, upper bound: 808.1583982
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577596, upper bound: 808.1577304
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577957, upper bound: 808.1577304
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945133, upper bound: 808.1954287
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945356, upper bound: 808.1945031
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945082, upper bound: 808.1945031
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945518, upper bound: 808.1945031
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1804807
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1813230, upper bound: 808.1806688
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1813573, upper bound: 808.1804101
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1813592, upper bound: 808.1804101
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1947097
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945064, upper bound: 808.1945031
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954188, upper bound: 808.1945031
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945132, upper bound: 808.1954287
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945223, upper bound: 808.1945031
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945129, upper bound: 808.1945031
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945129, upper bound: 808.1945031
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945132, upper bound: 808.1954287
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945195, upper bound: 808.1945031
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1947097
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945117, upper bound: 808.1945031
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954108, upper bound: 808.1945031
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1947097
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
time: 0.72 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1954108
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945117
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945195
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1954287, upper bound: 808.1945132
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945765
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945129
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945223
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1954287, upper bound: 808.1945132
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1813301
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1813562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1953873
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945144
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1813592
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1813573
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1806688, upper bound: 808.1813230
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1804807, upper bound: 808.1804101
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577734
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577596
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1583992, upper bound: 808.1577304
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577957
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577723
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577596
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1583982, upper bound: 808.1577413
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945106, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945261, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945095, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945846, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945106, upper bound: 808.1952006
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945261, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945088, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945088, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1954122, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945263
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945106, upper bound: 808.1954252
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945243, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945095, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945095, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945106, upper bound: 808.1954252
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945243, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945088, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945510, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945560
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1954115, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945621
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1948368
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1954115
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945510
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945088
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945243
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1954252, upper bound: 808.1945106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945846
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945095
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945243
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945106
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1953554
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945263, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1954122
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945517
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945088
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945261
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1952006, upper bound: 808.1945106
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945846
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945095
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945261
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945106
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577413, upper bound: 808.1583982
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577596, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577304, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1577957, upper bound: 808.1577304
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945133, upper bound: 808.1954287
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945356, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945082, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945518, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1804101, upper bound: 808.1804807
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1813230, upper bound: 808.1806688
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1813573, upper bound: 808.1804101
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1813592, upper bound: 808.1804101
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1947097
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945064, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1954188, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945132, upper bound: 808.1954287
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945223, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945129, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945129, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945132, upper bound: 808.1954287
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945195, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1947097
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945117, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1954108, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1947097
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.42
Output dim: 4, lower bound: -808.1945031, upper bound: 808.1945031

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800792, upper bound: 808.1800792
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=1011.34521484375
rel_dist={4: [-809.0067385931752, 809.0067385931752]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0036900, upper bound: 809.0037626
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0037626, upper bound: 809.0036900
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 4, lower bound: -809.0036900, upper bound: 809.0037626
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 4, lower bound: -809.0037626, upper bound: 809.0036900

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0036557, upper bound: 809.0035035
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0024435, upper bound: 809.0036513
time: 0.73 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0036513, upper bound: 809.0024435
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0024435, upper bound: 809.0036557
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -809.0036557, upper bound: 809.0035035
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -809.0024435, upper bound: 809.0036513
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -809.0036513, upper bound: 809.0024435
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 4, lower bound: -809.0024435, upper bound: 809.0036557

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813259
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811632
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813270
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813270, upper bound: 808.5810333
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5810333
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811570
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5812327
time: 0.68 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813259
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811632
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813270
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -808.5813270, upper bound: 808.5810333
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5810333
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811570
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5812327

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197399, upper bound: 808.5197665
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197823, upper bound: 808.5193886
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197357, upper bound: 808.5198311
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197374, upper bound: 808.5193886
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5197434
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5198400
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5198400, upper bound: 808.5188205
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197434, upper bound: 808.5188205
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197374
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5198311, upper bound: 808.5197357
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197823
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197665, upper bound: 808.5197399
time: 0.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5197399, upper bound: 808.5197665
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5197823, upper bound: 808.5193886
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5197357, upper bound: 808.5198311
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5197374, upper bound: 808.5193886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5197434
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5198400
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5198400, upper bound: 808.5188205
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5197434, upper bound: 808.5188205
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197374
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5198311, upper bound: 808.5197357
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197823
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.91
Output dim: 4, lower bound: -808.5197665, upper bound: 808.5197399

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015842, upper bound: 808.2013874
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015842, upper bound: 808.2022252
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022481, upper bound: 808.2014032
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022481, upper bound: 808.2014553
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015842, upper bound: 808.2022217
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015842, upper bound: 808.2022258
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017198, upper bound: 808.2014325
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022481, upper bound: 808.2014553
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014628, upper bound: 808.2013874
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014324, upper bound: 808.2021103
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022260, upper bound: 808.2013874
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021617, upper bound: 808.2014328
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014628, upper bound: 808.2022322
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014324, upper bound: 808.2022322
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014628, upper bound: 808.2014365
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017198, upper bound: 808.2014421
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014421, upper bound: 808.2017198
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014365, upper bound: 808.2022260
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022322, upper bound: 808.2014324
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022322, upper bound: 808.2014628
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014328, upper bound: 808.2021617
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013874, upper bound: 808.2022260
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021103, upper bound: 808.2014325
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013874, upper bound: 808.2014628
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014553, upper bound: 808.2022481
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014325, upper bound: 808.2022481
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022258, upper bound: 808.2015842
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022217, upper bound: 808.2015842
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014553, upper bound: 808.2022481
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014032, upper bound: 808.2022481
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022252, upper bound: 808.2015842
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013874, upper bound: 808.2015842
time: 0.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2015842, upper bound: 808.2013874
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2015842, upper bound: 808.2022252
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022481, upper bound: 808.2014032
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022481, upper bound: 808.2014553
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2015842, upper bound: 808.2022217
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2015842, upper bound: 808.2022258
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2017198, upper bound: 808.2014325
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022481, upper bound: 808.2014553
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014628, upper bound: 808.2013874
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014324, upper bound: 808.2021103
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022260, upper bound: 808.2013874
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2021617, upper bound: 808.2014328
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014628, upper bound: 808.2022322
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014324, upper bound: 808.2022322
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014628, upper bound: 808.2014365
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2017198, upper bound: 808.2014421
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014421, upper bound: 808.2017198
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014365, upper bound: 808.2022260
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022322, upper bound: 808.2014324
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022322, upper bound: 808.2014628
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014328, upper bound: 808.2021617
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2013874, upper bound: 808.2022260
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2021103, upper bound: 808.2014325
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2013874, upper bound: 808.2014628
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014553, upper bound: 808.2022481
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014325, upper bound: 808.2022481
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022258, upper bound: 808.2015842
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022217, upper bound: 808.2015842
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014553, upper bound: 808.2022481
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2014032, upper bound: 808.2022481
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2022252, upper bound: 808.2015842
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.96
Output dim: 4, lower bound: -808.2013874, upper bound: 808.2015842

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018941
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010640
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010720
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019169, upper bound: 808.2010662
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011241
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019169, upper bound: 808.2010662
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018906
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2012530, upper bound: 808.2010657
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018946
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2012530, upper bound: 808.2010657
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011014
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019169, upper bound: 808.2010662
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011241
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010662
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2010562
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011316, upper bound: 808.2010562
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2017769
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011013, upper bound: 808.2010562
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018948, upper bound: 808.2010562
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011009
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018305, upper bound: 808.2010562
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2019010
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2010562
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2019010
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2010562
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011053
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018948, upper bound: 808.2010562
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011109
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013886, upper bound: 808.2010562
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2013886
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011109, upper bound: 808.2010562
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018948
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011053, upper bound: 808.2010562
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011013
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019010, upper bound: 808.2010638
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011316
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019010, upper bound: 808.2010638
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018305
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018948
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011013
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017769, upper bound: 808.2010638
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011316
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010638
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010662, upper bound: 808.2019169
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011241, upper bound: 808.2010562
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010662, upper bound: 808.2019169
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011014, upper bound: 808.2010562
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010657, upper bound: 808.2012530
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018946, upper bound: 808.2010562
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010657, upper bound: 808.2012530
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018906, upper bound: 808.2010562
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010662, upper bound: 808.2019169
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011241, upper bound: 808.2010562
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010662, upper bound: 808.2019169
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010720, upper bound: 808.2010562
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010640, upper bound: 808.2012530
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018941, upper bound: 808.2010562
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2012530
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
time: 0.67 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018941
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010640
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010720
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2019169, upper bound: 808.2010662
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011241
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2019169, upper bound: 808.2010662
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018906
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2012530, upper bound: 808.2010657
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018946
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2012530, upper bound: 808.2010657
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011014
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2019169, upper bound: 808.2010662
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011241
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010662
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2011316, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2017769
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2011013, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2018948, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011009
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2018305, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2019010
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2019010
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010638, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011053
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2018948, upper bound: 808.2010562
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011109
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2013886, upper bound: 808.2010562
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2013886
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2011109, upper bound: 808.2010562
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018948
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2011053, upper bound: 808.2010562
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011013
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2019010, upper bound: 808.2010638
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011316
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2019010, upper bound: 808.2010638
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018305
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2018948
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2017769, upper bound: 808.2010638
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2011316
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010638
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010662, upper bound: 808.2019169
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2011241, upper bound: 808.2010562
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010662, upper bound: 808.2019169
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2011014, upper bound: 808.2010562
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010657, upper bound: 808.2012530
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2018946, upper bound: 808.2010562
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010657, upper bound: 808.2012530
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2018906, upper bound: 808.2010562
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010662, upper bound: 808.2019169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2011241, upper bound: 808.2010562
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010662, upper bound: 808.2019169
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010720, upper bound: 808.2010562
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010640, upper bound: 808.2012530
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2018941, upper bound: 808.2010562
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2012530
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.93
Output dim: 4, lower bound: -808.2010562, upper bound: 808.2010562

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953106
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944553
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1946420, upper bound: 808.1944476
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944633
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945150
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944659
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953116
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952808
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944511
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953121
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952808
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944564
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1946420, upper bound: 808.1944570
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944924
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944777
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944528
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944575
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945150
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944777
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1944476
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944540, upper bound: 808.1944476
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944693, upper bound: 808.1944476
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945223, upper bound: 808.1944476
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1951009
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944535, upper bound: 808.1944476
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944693, upper bound: 808.1944476
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944924, upper bound: 808.1944476
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953113, upper bound: 808.1944476
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944635
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1952469, upper bound: 808.1944476
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1953196
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944540, upper bound: 808.1944476
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944677, upper bound: 808.1944476
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945223, upper bound: 808.1944476
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1953196
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944535, upper bound: 808.1944476
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944677, upper bound: 808.1944476
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944918, upper bound: 808.1944476
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944963
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953112, upper bound: 808.1944476
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945019
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1947623, upper bound: 808.1944476
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1947623
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953112
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944963, upper bound: 808.1944476
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944918
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944677
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944535
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953196, upper bound: 808.1944551
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945223
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944677
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944540
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953196, upper bound: 808.1944551
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952469
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953113
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944924
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944693
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944535
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951009, upper bound: 808.1944551
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945223
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944693
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944540
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944551
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944777, upper bound: 808.1944476
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945150, upper bound: 808.1944476
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944528, upper bound: 808.1944476
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944777, upper bound: 808.1944476
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944924, upper bound: 808.1944476
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1946420
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944564, upper bound: 808.1944476
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1952808, upper bound: 808.1944476
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953121, upper bound: 808.1944476
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1946420
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944511, upper bound: 808.1944476
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1952808, upper bound: 808.1944476
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953116, upper bound: 808.1944476
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944659, upper bound: 808.1944476
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944633, upper bound: 808.1944476
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1946420
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944553, upper bound: 808.1944476
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953106, upper bound: 808.1944476
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1946420
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
time: 0.65 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953106
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944553
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1946420, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944633
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945150
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944659
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953116
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952808
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944511
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953121
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952808
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944564
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1946420, upper bound: 808.1944570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944924
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944777
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944528
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944575
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945150
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944777
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944540, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944693, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1945223, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1951009
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944535, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944693, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944924, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953113, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944635
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1952469, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1953196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944540, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944677, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1945223, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1953196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944535, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944677, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944918, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953112, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945019
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1947623, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1947623
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953112
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944963, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944918
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953196, upper bound: 808.1944551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945223
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944540
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953196, upper bound: 808.1944551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944924
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944693
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944535
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1951009, upper bound: 808.1944551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944693
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944540
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944551
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944777, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1945150, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944528, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944777, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944924, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1946420
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944564, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1952808, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953121, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1946420
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944511, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1952808, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953116, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944659, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944633, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1946420
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944553, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1953106, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1946420
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1808424
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800380
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1802301, upper bound: 808.1800298
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
time: 0.71 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1808424
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1802301, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.23
Output dim: 4, lower bound: -808.1800298, upper bound: 808.1800298
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944633
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945150
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944659
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953116
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952808
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944511
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953121
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952808
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944564
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1946420, upper bound: 808.1944570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944924
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944777
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944528
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944575
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945150
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944777
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953300, upper bound: 808.1944575
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944540, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944693, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1945223, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1951009
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944535, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944693, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944924, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953113, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944635
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1952469, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1953196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944540, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944677, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1945223, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944551, upper bound: 808.1953196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944535, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944677, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944918, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944963
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953112, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945019
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1947623, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1947623
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953112
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944963, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944918
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944535
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953196, upper bound: 808.1944551
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945223
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944677
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944540
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953196, upper bound: 808.1944551
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1952469
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1953113
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944924
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944693
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944535
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1951009, upper bound: 808.1944551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1945223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944693
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944540
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944551
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944777, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1945150, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944528, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944777, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944924, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1946420
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944564, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1952808, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953121, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1946420
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944511, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1952808, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953116, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944659, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944570, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944575, upper bound: 808.1953300
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944633, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1946420
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944553, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1953106, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1946420
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.23
Output dim: 4, lower bound: -808.1944476, upper bound: 808.1944476
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=1011.34521484375
rel_dist={4: [-809.0065995903992, 809.0065995903992]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0034812, upper bound: 809.0035339
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0035339, upper bound: 809.0034812
time: 0.91 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 4, lower bound: -809.0034812, upper bound: 809.0035339
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 4, lower bound: -809.0035339, upper bound: 809.0034812

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0034048, upper bound: 809.0032960
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0022081, upper bound: 809.0034016
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0034016, upper bound: 809.0022081
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0032960, upper bound: 809.0034048
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -809.0034048, upper bound: 809.0032960
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -809.0022081, upper bound: 809.0034016
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -809.0034016, upper bound: 809.0022081
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 4, lower bound: -809.0032960, upper bound: 809.0034048

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811570, upper bound: 808.5813207
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811632
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813250
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813250, upper bound: 808.5810333
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5810333
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813207, upper bound: 808.5811570
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811862, upper bound: 808.5812327
time: 0.63 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -808.5811570, upper bound: 808.5813207
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811632
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813250
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -808.5813250, upper bound: 808.5810333
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5810333
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -808.5813207, upper bound: 808.5811570
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 4, lower bound: -808.5811862, upper bound: 808.5812327

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197399, upper bound: 808.5197621
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197823, upper bound: 808.5193886
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197357, upper bound: 808.5198194
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197374, upper bound: 808.5193886
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188191, upper bound: 808.5197434
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188191, upper bound: 808.5198281
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5198281, upper bound: 808.5188191
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197434, upper bound: 808.5188191
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197374
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5198194, upper bound: 808.5197357
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197823
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197621, upper bound: 808.5197399
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5197399, upper bound: 808.5197621
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5197823, upper bound: 808.5193886
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5197357, upper bound: 808.5198194
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5197374, upper bound: 808.5193886
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5188191, upper bound: 808.5197434
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5188191, upper bound: 808.5198281
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5198281, upper bound: 808.5188191
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5197434, upper bound: 808.5188191
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197374
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5198194, upper bound: 808.5197357
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197823
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.95
Output dim: 4, lower bound: -808.5197621, upper bound: 808.5197399

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5185700, upper bound: 808.5193846
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193622, upper bound: 808.5185622
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5185883, upper bound: 808.5190112
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5194046, upper bound: 808.5185386
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5185230, upper bound: 808.5194420
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193580, upper bound: 808.5188891
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5183252, upper bound: 808.5190112
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193598, upper bound: 808.5185459
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5183430, upper bound: 808.5193658
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5184417, upper bound: 808.5183252
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5183252, upper bound: 808.5193401
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5192325, upper bound: 808.5185319
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5183430, upper bound: 808.5194506
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5184417, upper bound: 808.5188880
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5183252, upper bound: 808.5193401
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5192325, upper bound: 808.5185641
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5185641, upper bound: 808.5192325
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193401, upper bound: 808.5183252
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188880, upper bound: 808.5184417
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5194506, upper bound: 808.5183430
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5185319, upper bound: 808.5192325
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193401, upper bound: 808.5183252
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5183252, upper bound: 808.5184417
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193658, upper bound: 808.5183430
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5185459, upper bound: 808.5193598
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5190112, upper bound: 808.5183252
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188891, upper bound: 808.5193580
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5194420, upper bound: 808.5185230
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5185386, upper bound: 808.5194046
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5190112, upper bound: 808.5185883
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5185622, upper bound: 808.5193622
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193846, upper bound: 808.5185700
time: 0.59 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5185700, upper bound: 808.5193846
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5193622, upper bound: 808.5185622
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5185883, upper bound: 808.5190112
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5194046, upper bound: 808.5185386
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5185230, upper bound: 808.5194420
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5193580, upper bound: 808.5188891
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5183252, upper bound: 808.5190112
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5193598, upper bound: 808.5185459
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5183430, upper bound: 808.5193658
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5184417, upper bound: 808.5183252
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5183252, upper bound: 808.5193401
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5192325, upper bound: 808.5185319
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5183430, upper bound: 808.5194506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5184417, upper bound: 808.5188880
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5183252, upper bound: 808.5193401
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5192325, upper bound: 808.5185641
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5185641, upper bound: 808.5192325
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5193401, upper bound: 808.5183252
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5188880, upper bound: 808.5184417
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5194506, upper bound: 808.5183430
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5185319, upper bound: 808.5192325
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5193401, upper bound: 808.5183252
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5183252, upper bound: 808.5184417
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5193658, upper bound: 808.5183430
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5185459, upper bound: 808.5193598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5190112, upper bound: 808.5183252
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5188891, upper bound: 808.5193580
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5194420, upper bound: 808.5185230
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5185386, upper bound: 808.5194046
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5190112, upper bound: 808.5185883
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5185622, upper bound: 808.5193622
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.00
Output dim: 4, lower bound: -808.5193846, upper bound: 808.5185700

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017535
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011659, upper bound: 808.2009721
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009797
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009876
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010389
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017764, upper bound: 808.2009819
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009819
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017500
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017541
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009814
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009814
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010165
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010389
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009819
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009819
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2009721
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2016364
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010463, upper bound: 808.2009721
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010164, upper bound: 808.2009721
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010155
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017543, upper bound: 808.2009721
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017605
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2017605
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2009721
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2009721
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010204
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010259
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017543, upper bound: 808.2009721
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2012867
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017543
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010204, upper bound: 808.2009721
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010163
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010463
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017605, upper bound: 808.2009795
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009795
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2016900
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017543
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010164
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010463
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016364, upper bound: 808.2009795
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009795
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2017764
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2017764
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2010389, upper bound: 808.2009721
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2009721
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009797, upper bound: 808.2011659
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009814, upper bound: 808.2011659
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017541, upper bound: 808.2009721
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017500, upper bound: 808.2009721
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2017764
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2017764
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2009721
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009876, upper bound: 808.2009721
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009797, upper bound: 808.2011659
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2011659
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017535, upper bound: 808.2009721
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
time: 0.66 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2011659, upper bound: 808.2009721
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009797
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2017764, upper bound: 808.2009819
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009819
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017500
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017541
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009814
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009814
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010165
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010389
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009819
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009819
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2016364
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2010463, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2010164, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010155
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2017543, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017605
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2017605
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009795, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010204
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010259
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2017543, upper bound: 808.2009721
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2012867
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017543
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2010204, upper bound: 808.2009721
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010163
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010463
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2017605, upper bound: 808.2009795
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009795
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2016900
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2017543
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010164
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2010463
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2016364, upper bound: 808.2009795
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009795
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2017764
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2017764
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2010389, upper bound: 808.2009721
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2009721
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009797, upper bound: 808.2011659
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009814, upper bound: 808.2011659
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2017541, upper bound: 808.2009721
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2017500, upper bound: 808.2009721
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2017764
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2017764
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009819, upper bound: 808.2009721
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009876, upper bound: 808.2009721
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009797, upper bound: 808.2011659
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2011659
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2017535, upper bound: 808.2009721
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.20
Output dim: 4, lower bound: -808.2009721, upper bound: 808.2009721

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1951709
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945513, upper bound: 808.1943576
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943652
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945513, upper bound: 808.1943576
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943728
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1944244
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943753
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951904, upper bound: 808.1943671
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943667
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951904, upper bound: 808.1943671
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1951715
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1951401
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1951725
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1951401
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943605
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945513, upper bound: 808.1943668
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943662
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945513, upper bound: 808.1943668
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1944019
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943871
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1944244
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943871
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943623
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951904, upper bound: 808.1943671
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943667
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951904, upper bound: 808.1943671
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943646, upper bound: 808.1943576
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943635, upper bound: 808.1943576
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943646, upper bound: 808.1949613
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943629, upper bound: 808.1943576
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943787, upper bound: 808.1943576
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944318, upper bound: 808.1943576
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943787, upper bound: 808.1943576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944018, upper bound: 808.1943576
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943729
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951716, upper bound: 808.1943576
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951073, upper bound: 808.1943576
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943646, upper bound: 808.1951800
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943635, upper bound: 808.1943576
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943646, upper bound: 808.1951800
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943629, upper bound: 808.1943576
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943771, upper bound: 808.1943576
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944318, upper bound: 808.1943576
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943771, upper bound: 808.1943576
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944013, upper bound: 808.1943576
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1944058
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1943576, upper bound: 808.1943576
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.71 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=1011.34521484375
rel_dist={4: [-809.0063734281499, 809.00637342815]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1130.35 seconds
