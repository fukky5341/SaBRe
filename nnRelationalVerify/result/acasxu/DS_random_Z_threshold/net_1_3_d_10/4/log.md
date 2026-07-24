## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 807.3886655422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953)
1: (-373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194)
2: (-542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741)
3: (-209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672)
4: (-604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.80 + 1.73 = 2.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -809.0066789, upper bound: 809.0066789

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0056894, upper bound: 809.0061983
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0061983, upper bound: 809.0056894
time: 0.60 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.25 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -809.0056894, upper bound: 809.0061983
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.25
Output dim: 4, lower bound: -809.0061983, upper bound: 809.0056894

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9871400, upper bound: 808.9876528
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9870815, upper bound: 808.9876515
time: 0.47 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5889273, upper bound: 808.5882263
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5882263, upper bound: 808.5885073
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.02 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 4, lower bound: -808.9871400, upper bound: 808.9876528
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 4, lower bound: -808.9870815, upper bound: 808.9876515
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 4, lower bound: -808.5889273, upper bound: 808.5882263
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 4, lower bound: -808.5882263, upper bound: 808.5885073

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9842104, upper bound: 808.9846951
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9841168, upper bound: 808.9846656
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0590853, upper bound: 806.0591799
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0590853, upper bound: 806.0591799
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0592452, upper bound: 806.0591998
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.0592452, upper bound: 806.0591998
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3775411, upper bound: 808.3769242
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3771048, upper bound: 808.3770287
time: 0.71 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.10 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 4, lower bound: -808.9842104, upper bound: 808.9846951
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 4, lower bound: -808.9841168, upper bound: 808.9846656
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 4, lower bound: -806.0590853, upper bound: 806.0591799
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 4, lower bound: -806.0590853, upper bound: 806.0591799
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 4, lower bound: -806.0592452, upper bound: 806.0591998
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.10
Output dim: 4, lower bound: -806.0592452, upper bound: 806.0591998
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 4, lower bound: -808.3775411, upper bound: 808.3769242
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.10
Output dim: 4, lower bound: -808.3771048, upper bound: 808.3770287

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9779024, upper bound: 808.9791436
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9785956, upper bound: 808.9777948
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9832334, upper bound: 808.9846656
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9832334, upper bound: 808.9846419
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3643930, upper bound: 808.3632050
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3631532, upper bound: 808.3637580
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3761381, upper bound: 808.3761513
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3761513, upper bound: 808.3761381
time: 0.69 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 4, lower bound: -808.9779024, upper bound: 808.9791436
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 4, lower bound: -808.9785956, upper bound: 808.9777948
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 4, lower bound: -808.9832334, upper bound: 808.9846656
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 4, lower bound: -808.9832334, upper bound: 808.9846419
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 4, lower bound: -808.3643930, upper bound: 808.3632050
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 4, lower bound: -808.3631532, upper bound: 808.3637580
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 4, lower bound: -808.3761381, upper bound: 808.3761513
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.51
Output dim: 4, lower bound: -808.3761513, upper bound: 808.3761381

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9776975, upper bound: 808.9790103
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9775252, upper bound: 808.9790146
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9775617, upper bound: 808.9774591
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9777345, upper bound: 808.9774591
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9541525, upper bound: 808.9554977
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9541756, upper bound: 808.9555013
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1450182, upper bound: 807.1448998
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1450130, upper bound: 807.1450278
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3250620, upper bound: 808.3251033
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3262913, upper bound: 808.3250587
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3250515, upper bound: 808.3256563
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3250515, upper bound: 808.3250587
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3760171, upper bound: 808.3760349
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3760182, upper bound: 808.3760165
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2176923, upper bound: 808.2176459
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2183766, upper bound: 808.2176459
time: 0.56 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.9776975, upper bound: 808.9790103
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.9775252, upper bound: 808.9790146
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.9775617, upper bound: 808.9774591
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.9777345, upper bound: 808.9774591
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.9541525, upper bound: 808.9554977
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.9541756, upper bound: 808.9555013
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 4, lower bound: -807.1450182, upper bound: 807.1448998
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.00
Output dim: 4, lower bound: -807.1450130, upper bound: 807.1450278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.3250620, upper bound: 808.3251033
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.3262913, upper bound: 808.3250587
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.3250515, upper bound: 808.3256563
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.3250515, upper bound: 808.3250587
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.3760171, upper bound: 808.3760349
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.3760182, upper bound: 808.3760165
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.2176923, upper bound: 808.2176459
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.00
Output dim: 4, lower bound: -808.2183766, upper bound: 808.2176459

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9780055, upper bound: 808.9790103
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9777413, upper bound: 808.9775745
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1413457, upper bound: 807.1413410
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -807.1413453, upper bound: 807.1414650
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9453274
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9454494, upper bound: 808.9453297
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9452318
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9455607, upper bound: 808.9453422
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8140759, upper bound: 806.8140759
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8140759, upper bound: 806.8140787
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9538267, upper bound: 808.9552426
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9538266, upper bound: 808.9552430
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3250515, upper bound: 808.3251033
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3250620, upper bound: 808.3250515
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578501, upper bound: 808.1578238
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578259
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1581573
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578302
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3249466, upper bound: 808.3249531
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3249466, upper bound: 808.3249542
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3280514, upper bound: 808.3278615
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3278431, upper bound: 808.3278431
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3760019, upper bound: 808.3759979
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3759979, upper bound: 808.3759979
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753422, upper bound: 808.1753422
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753543, upper bound: 808.1753422
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2175623, upper bound: 808.2175623
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2183761, upper bound: 808.2175623
time: 0.54 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.9780055, upper bound: 808.9790103
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.9777413, upper bound: 808.9775745
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.91
Output dim: 4, lower bound: -807.1413457, upper bound: 807.1413410
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.91
Output dim: 4, lower bound: -807.1413453, upper bound: 807.1414650
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9453274
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.9454494, upper bound: 808.9453297
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.9452318, upper bound: 808.9452318
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.9455607, upper bound: 808.9453422
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.91
Output dim: 4, lower bound: -806.8140759, upper bound: 806.8140759
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.91
Output dim: 4, lower bound: -806.8140759, upper bound: 806.8140787
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.9538267, upper bound: 808.9552426
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.9538266, upper bound: 808.9552430
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.3250515, upper bound: 808.3251033
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.3250620, upper bound: 808.3250515
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.1578501, upper bound: 808.1578238
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578259
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1581573
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578302
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.3249466, upper bound: 808.3249531
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.3249466, upper bound: 808.3249542
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.3280514, upper bound: 808.3278615
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.3278431, upper bound: 808.3278431
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.3760019, upper bound: 808.3759979
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.3759979, upper bound: 808.3759979
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.1753422, upper bound: 808.1753422
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.1753543, upper bound: 808.1753422
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.2175623, upper bound: 808.2175623
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.91
Output dim: 4, lower bound: -808.2183761, upper bound: 808.2175623

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9450600, upper bound: 808.9464282
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9453190, upper bound: 808.9464897
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.1470165, upper bound: 806.1470157
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.1470083, upper bound: 806.1470157
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9450549
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9450807
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465418
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465428
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465384
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465384
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9452559, upper bound: 808.9450810
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.9453283, upper bound: 808.9449834
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8138688, upper bound: 806.8138688
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.8138688, upper bound: 806.8138724
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247477, upper bound: 808.3247294
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247477, upper bound: 808.3256834
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248217
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248691
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248165
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248165
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578238
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578501, upper bound: 808.1578238
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578259
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1585964, upper bound: 808.1578238
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574525
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578302
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578238
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3246649, upper bound: 808.3246724
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3246649, upper bound: 808.3246649
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577909
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752926, upper bound: 808.1753103
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752926, upper bound: 808.1753110
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3247475
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3247475
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3283348, upper bound: 808.3278246
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3282494, upper bound: 808.3278246
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753286, upper bound: 808.1753286
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753286, upper bound: 808.1753286
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753412, upper bound: 808.1753286
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1753407, upper bound: 808.1753286
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2175636, upper bound: 808.2175508
time: 0.66 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.80 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.9450600, upper bound: 808.9464282
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.9453190, upper bound: 808.9464897
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 4, lower bound: -806.1470165, upper bound: 806.1470157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 4, lower bound: -806.1470083, upper bound: 806.1470157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9450549
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.9449834, upper bound: 808.9450807
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465418
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465428
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465384
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 4, lower bound: -806.7465385, upper bound: 806.7465384
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.9452559, upper bound: 808.9450810
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.9453283, upper bound: 808.9449834
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 4, lower bound: -806.8138688, upper bound: 806.8138688
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.80
Output dim: 4, lower bound: -806.8138688, upper bound: 806.8138724
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3247477, upper bound: 808.3247294
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3247477, upper bound: 808.3256834
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248217
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248691
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248165
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3248165, upper bound: 808.3248165
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578238
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1578501, upper bound: 808.1578238
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578259
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1585964, upper bound: 808.1578238
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574525
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578302
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1578238, upper bound: 808.1578238
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3246649, upper bound: 808.3246724
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3246649, upper bound: 808.3246649
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577909
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1752926, upper bound: 808.1753103
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1752926, upper bound: 808.1753110
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3247475
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3247475, upper bound: 808.3247475
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3283348, upper bound: 808.3278246
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.3282494, upper bound: 808.3278246
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1753286, upper bound: 808.1753286
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1753286, upper bound: 808.1753286
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1753412, upper bound: 808.1753286
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.1753407, upper bound: 808.1753286
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.2175508, upper bound: 808.2175508
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.80
Output dim: 4, lower bound: -808.2175636, upper bound: 808.2175508

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800662
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1808959
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1809234
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1809240
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464657
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2006019
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2006108
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464657
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800771, upper bound: 808.1800662
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800662
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800771, upper bound: 808.1809316
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800810
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577867
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577884
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245459
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3246892, upper bound: 808.3245349
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574848, upper bound: 808.1574425
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574507
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1582784, upper bound: 808.1577865
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574525
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577955
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577950
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574351
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577909
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1576229, upper bound: 808.1574425
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574525
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574425
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580959, upper bound: 808.1574530
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005740, upper bound: 808.2005610
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3253195, upper bound: 808.3245349
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.3245777, upper bound: 808.3245349
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574351, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574705, upper bound: 808.1574280
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574465, upper bound: 808.1574280
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574440, upper bound: 808.1574280
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574383, upper bound: 808.1574280
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2012295, upper bound: 808.2005610
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2005900, upper bound: 808.2005610
time: 0.67 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800662
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1808959
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1809234
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1809240
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464657
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2006019
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2006108
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464657
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.54
Output dim: 4, lower bound: -806.7464646, upper bound: 806.7464646
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1800771, upper bound: 808.1800662
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800662
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1800771, upper bound: 808.1809316
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800810
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577867
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577884
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245459
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3246892, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574848, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574507
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1582784, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574525
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577955
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577950
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574351
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577909
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1577865, upper bound: 808.1577865
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1576229, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574425, upper bound: 808.1574525
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574425
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1580959, upper bound: 808.1574530
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3245349, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005740, upper bound: 808.2005610
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3253195, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.3245777, upper bound: 808.3245349
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574351, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574705, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574465, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574440, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574383, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.1752811, upper bound: 808.1752811
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005610, upper bound: 808.2005610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2012295, upper bound: 808.2005610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.54
Output dim: 4, lower bound: -808.2005900, upper bound: 808.2005610

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1579203
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580966
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580964
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580966
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580964
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574690
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574748
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574778
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574362, upper bound: 808.1574280
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580843
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574362, upper bound: 808.1574280
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574320
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574339
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574379
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574386
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1574280
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1574280
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574345
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574362
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580964, upper bound: 808.1574280
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574380
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574379
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574384
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574386
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574351
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574362
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1576096, upper bound: 808.1574280
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574380
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574280
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574385
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580959, upper bound: 808.1574385
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574384, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1575345, upper bound: 808.1574280
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574280
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574690, upper bound: 808.1574280
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1577282, upper bound: 808.1574280
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574482, upper bound: 808.1574280
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574482, upper bound: 808.1574280
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574753, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574351, upper bound: 808.1574280
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574627, upper bound: 808.1574280
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574386, upper bound: 808.1574280
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574440, upper bound: 808.1574280
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574384, upper bound: 808.1574280
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1579502, upper bound: 808.1574280
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1576967, upper bound: 808.1574280
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574571, upper bound: 808.1574280
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
time: 0.68 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1579203
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580966
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580964
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580966
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580964
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574690
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574748
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574778
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574362, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574362, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574320
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574339
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574379
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574386
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574345
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574362
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1580964, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574380
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574379
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574384
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574386
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574351
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574362
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1576096, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574380
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574385
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1580959, upper bound: 808.1574385
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574384, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1575345, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574690, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1577282, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574482, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574482, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574753, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574351, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574627, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574386, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574440, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574384, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1579502, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1576967, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574571, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.46
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573815, upper bound: 808.1574280
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573815
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### Candidate
type: DSZ, layer: 1, pos: 23

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 33
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 20

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1579203
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 35

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 18
type: DSZ, layer: 3, pos: 35
type: DSZ, layer: 3, pos: 10
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 12
type: DSZ, layer: 3, pos: 8
type: DSZ, layer: 3, pos: 22
type: DSZ, layer: 3, pos: 43
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 25
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 31
type: DSZ, layer: 3, pos: 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 30
type: DSZ, layer: 3, pos: 16
type: DSZ, layer: 3, pos: 9
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 17
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 20
type: DSZ, layer: 3, pos: 7
type: DSZ, layer: 3, pos: 33

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1561752, upper bound: 808.1561847
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1561847, upper bound: 808.1561752
time: 0.65 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 2.63 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 4, lower bound: -808.1573815, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573815
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1573777
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 4, lower bound: -808.1573777, upper bound: 808.1579203
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 4, lower bound: -808.1561752, upper bound: 808.1561847
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.63
Output dim: 4, lower bound: -808.1561847, upper bound: 808.1561752
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580966
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580964
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580966
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580964
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574690
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574748
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574778
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574362, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1580843
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574362, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574320
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574339
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574379
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574386
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574385, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574345
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574362
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1580964, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574380
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574379
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574384
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574386
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574351
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574362
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1576096, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574380
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574385
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1580959, upper bound: 808.1574385
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574384, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1575345, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1580968, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574690, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1577282, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574482, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574482, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574753, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574351, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574627, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574386, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574440, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574384, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1579502, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1576967, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1580966, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574571, upper bound: 808.1574280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.63
Output dim: 4, lower bound: -808.1574280, upper bound: 808.1574280

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.53 + 418.50 = 421.03 seconds
