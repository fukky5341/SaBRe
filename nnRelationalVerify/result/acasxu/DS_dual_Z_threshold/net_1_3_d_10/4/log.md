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
execution time: IAR + RelationalAnalysis = 1.66 + 1.78 = 3.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -809.0066789, upper bound: 809.0066789

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0037698, upper bound: 809.0038429
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0038429, upper bound: 809.0037698
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.49 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 4, lower bound: -809.0037698, upper bound: 809.0038429
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 4, lower bound: -809.0038429, upper bound: 809.0037698

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0024899, upper bound: 809.0035866
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0024899, upper bound: 809.0037415
time: 0.56 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0037415, upper bound: 809.0024899
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -809.0035866, upper bound: 809.0037387
time: 0.67 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.11 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 4, lower bound: -809.0024899, upper bound: 809.0035866
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 4, lower bound: -809.0024899, upper bound: 809.0037415
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 4, lower bound: -809.0037415, upper bound: 809.0024899
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 4, lower bound: -809.0035866, upper bound: 809.0037387

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811570, upper bound: 808.5813259
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811632
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813270
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5810333
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5810333
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5813259, upper bound: 808.5811570
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5811862, upper bound: 808.5812327
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5811862
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -808.5811570, upper bound: 808.5813259
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5811632
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -808.5810333, upper bound: 808.5813270
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -808.5812327, upper bound: 808.5810333
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -808.5811632, upper bound: 808.5810333
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -808.5813259, upper bound: 808.5811570
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 4, lower bound: -808.5811862, upper bound: 808.5812327

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197399, upper bound: 808.5197665
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197823, upper bound: 808.5193886
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197357, upper bound: 808.5198311
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197374, upper bound: 808.5193886
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5197434
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5198400
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5196102
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5198400, upper bound: 808.5188205
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197434, upper bound: 808.5188205
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197374
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5198311, upper bound: 808.5197357
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197823
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.5197665, upper bound: 808.5197399
time: 0.71 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5197399, upper bound: 808.5197665
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5197823, upper bound: 808.5193886
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5197357, upper bound: 808.5198311
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5197374, upper bound: 808.5193886
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5197434
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5188205, upper bound: 808.5198400
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5196102, upper bound: 808.5197177
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5196102
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5198400, upper bound: 808.5188205
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5197177, upper bound: 808.5196102
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5197434, upper bound: 808.5188205
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197374
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5198311, upper bound: 808.5197357
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5193886, upper bound: 808.5197823
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.14
Output dim: 4, lower bound: -808.5197665, upper bound: 808.5197399

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016324, upper bound: 808.2014342
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016324, upper bound: 808.2023046
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023287, upper bound: 808.2014500
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023287, upper bound: 808.2015026
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016324, upper bound: 808.2023018
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2016324, upper bound: 808.2023053
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023054, upper bound: 808.2014796
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017718, upper bound: 808.2015026
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015101, upper bound: 808.2014342
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014795, upper bound: 808.2021897
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023054, upper bound: 808.2014342
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2022426, upper bound: 808.2014801
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015101, upper bound: 808.2023125
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014795, upper bound: 808.2023125
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023054, upper bound: 808.2014836
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2017718, upper bound: 808.2014893
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014893, upper bound: 808.2017718
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014342, upper bound: 808.2023054
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023125, upper bound: 808.2014795
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023125, upper bound: 808.2015101
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014801, upper bound: 808.2022426
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014342, upper bound: 808.2023054
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2021897, upper bound: 808.2014796
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014342, upper bound: 808.2015101
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015026, upper bound: 808.2023287
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014500, upper bound: 808.2023287
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023053, upper bound: 808.2016324
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023018, upper bound: 808.2016324
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2015026, upper bound: 808.2023287
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014500, upper bound: 808.2023287
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2023046, upper bound: 808.2016324
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014342, upper bound: 808.2016324
time: 0.78 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2016324, upper bound: 808.2014342
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2016324, upper bound: 808.2023046
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023287, upper bound: 808.2014500
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023287, upper bound: 808.2015026
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2016324, upper bound: 808.2023018
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2016324, upper bound: 808.2023053
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023054, upper bound: 808.2014796
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2017718, upper bound: 808.2015026
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2015101, upper bound: 808.2014342
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014795, upper bound: 808.2021897
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023054, upper bound: 808.2014342
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2022426, upper bound: 808.2014801
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2015101, upper bound: 808.2023125
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014795, upper bound: 808.2023125
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023054, upper bound: 808.2014836
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2017718, upper bound: 808.2014893
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014893, upper bound: 808.2017718
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014342, upper bound: 808.2023054
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023125, upper bound: 808.2014795
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023125, upper bound: 808.2015101
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014801, upper bound: 808.2022426
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014342, upper bound: 808.2023054
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2021897, upper bound: 808.2014796
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014342, upper bound: 808.2015101
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2015026, upper bound: 808.2023287
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014500, upper bound: 808.2023287
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023053, upper bound: 808.2016324
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023018, upper bound: 808.2016324
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2015026, upper bound: 808.2023287
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014500, upper bound: 808.2023287
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2023046, upper bound: 808.2016324
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 4, lower bound: -808.2014342, upper bound: 808.2016324

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011031
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013012, upper bound: 808.2011031
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019735
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013012, upper bound: 808.2011108
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011189
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019975, upper bound: 808.2011130
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011714
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019975, upper bound: 808.2011130
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019706
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013012, upper bound: 808.2011126
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019741
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2013012, upper bound: 808.2011126
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011485
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019975, upper bound: 808.2011130
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011714
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019975, upper bound: 808.2011130
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011106, upper bound: 808.2011031
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011790, upper bound: 808.2011031
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011106, upper bound: 808.2018563
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011485, upper bound: 808.2011031
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011031
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019742, upper bound: 808.2011031
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011483
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019114, upper bound: 808.2011031
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011106, upper bound: 808.2019813
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011790, upper bound: 808.2011031
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011106, upper bound: 808.2019813
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011484, upper bound: 808.2011031
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011525
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019742, upper bound: 808.2011031
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011581
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2014407, upper bound: 808.2011031
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2014407
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011581, upper bound: 808.2011031
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019742
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011525, upper bound: 808.2011031
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011484
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019813, upper bound: 808.2011106
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011790
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019813, upper bound: 808.2011106
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019114
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011483, upper bound: 808.2011031
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019742
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011031
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011485
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2018563, upper bound: 808.2011106
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011790
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011106
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011130, upper bound: 808.2019975
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011714, upper bound: 808.2011031
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011130, upper bound: 808.2019975
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011485, upper bound: 808.2011031
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011126, upper bound: 808.2013012
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019741, upper bound: 808.2011031
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011126, upper bound: 808.2013012
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019706, upper bound: 808.2011031
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011130, upper bound: 808.2019975
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011714, upper bound: 808.2011031
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011130, upper bound: 808.2019975
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011189, upper bound: 808.2011031
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011108, upper bound: 808.2013012
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2019735, upper bound: 808.2011031
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2013012
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011031
time: 0.72 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011031
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2013012, upper bound: 808.2011031
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019735
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2013012, upper bound: 808.2011108
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011189
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019975, upper bound: 808.2011130
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011714
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019975, upper bound: 808.2011130
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019706
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2013012, upper bound: 808.2011126
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019741
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2013012, upper bound: 808.2011126
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011485
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019975, upper bound: 808.2011130
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011714
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019975, upper bound: 808.2011130
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011106, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011790, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011106, upper bound: 808.2018563
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011485, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019742, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011483
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019114, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011106, upper bound: 808.2019813
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011790, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011106, upper bound: 808.2019813
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011484, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011525
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019742, upper bound: 808.2011031
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011581
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2014407, upper bound: 808.2011031
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2014407
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011581, upper bound: 808.2011031
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019742
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011525, upper bound: 808.2011031
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011484
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019813, upper bound: 808.2011106
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011790
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019813, upper bound: 808.2011106
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019114
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011483, upper bound: 808.2011031
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2019742
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011031
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011485
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2018563, upper bound: 808.2011106
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011790
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011106
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011130, upper bound: 808.2019975
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011714, upper bound: 808.2011031
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011130, upper bound: 808.2019975
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011485, upper bound: 808.2011031
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011126, upper bound: 808.2013012
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019741, upper bound: 808.2011031
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011126, upper bound: 808.2013012
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019706, upper bound: 808.2011031
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011130, upper bound: 808.2019975
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011714, upper bound: 808.2011031
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011130, upper bound: 808.2019975
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011189, upper bound: 808.2011031
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011108, upper bound: 808.2013012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2019735, upper bound: 808.2011031
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2013012
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 4, lower bound: -808.2011031, upper bound: 808.2011031

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953864
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944980
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1946924, upper bound: 808.1944895
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945059
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944996
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945629
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945087
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944993
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944996
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953919
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953604
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944928
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945004
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953924
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953604
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944991
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945004
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945382
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945220
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944946
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944997
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945629
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945220
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944993
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954045, upper bound: 808.1944997
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1944895
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1944895
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945705, upper bound: 808.1944895
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1951762
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945125, upper bound: 808.1944895
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945381, upper bound: 808.1944895
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945117
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953272, upper bound: 808.1944895
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1954001
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945107, upper bound: 808.1944895
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1954001
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945107, upper bound: 808.1944895
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945424
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945485
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1948173, upper bound: 808.1944895
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1948173
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953871
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945424, upper bound: 808.1944895
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945374
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945107
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944952
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954001, upper bound: 808.1944970
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945705
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945107
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944958
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1954001, upper bound: 808.1944970
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953272
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953877
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945381
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945125
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944952
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1951762, upper bound: 808.1944970
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945705
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945125
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944958
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944970
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944997, upper bound: 808.1954045
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944993, upper bound: 808.1944895
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945220, upper bound: 808.1944895
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945629, upper bound: 808.1944895
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944997, upper bound: 808.1954045
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944946, upper bound: 808.1944895
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945220, upper bound: 808.1944895
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945382, upper bound: 808.1944895
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945004, upper bound: 808.1946924
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944991, upper bound: 808.1944895
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953604, upper bound: 808.1944895
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953924, upper bound: 808.1944895
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945004, upper bound: 808.1946924
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944928, upper bound: 808.1944895
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953604, upper bound: 808.1944895
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944928, upper bound: 808.1944895
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1954045
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944993, upper bound: 808.1944895
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945087, upper bound: 808.1944895
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945629, upper bound: 808.1944895
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944996, upper bound: 808.1954045
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1945059, upper bound: 808.1944895
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1946924
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944980, upper bound: 808.1944895
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1953864, upper bound: 808.1944895
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1946924
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
time: 0.72 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953864
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944980
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1946924, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945059
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944996
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945629
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945087
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944993
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944996
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953604
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944928
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945004
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953924
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953604
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944991
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945004
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945382
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944946
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944997
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945629
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944993
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1954045, upper bound: 808.1944997
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945705, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1951762
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945125, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945381, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945117
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1953272, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1954001
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945107, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1954001
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945107, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945424
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945485
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1948173, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1948173
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953871
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945424, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945374
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945107
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1954001, upper bound: 808.1944970
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945705
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945107
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944958
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1954001, upper bound: 808.1944970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953272
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953877
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945381
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945125
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1951762, upper bound: 808.1944970
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945705
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945125
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944958
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944997, upper bound: 808.1954045
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944993, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945220, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945629, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944997, upper bound: 808.1954045
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944946, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945220, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945382, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945004, upper bound: 808.1946924
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944991, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1953604, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1953924, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945004, upper bound: 808.1946924
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944928, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1953604, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944928, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1954045
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944993, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945087, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945629, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944996, upper bound: 808.1954045
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1945059, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1946924
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944980, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1953864, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1946924
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.70
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -465.2375793, 388.7125854, -465.2375793, 388.7125854, -853.9501953, 853.9501953
1: -373.0900269, 376.9524841, -373.0900269, 376.9524841, -750.0424194, 750.0424194
2: -542.3095093, 411.3765564, -542.3095093, 411.3765564, -953.6859131, 953.6859741
3: -209.7973175, 530.6040649, -209.7973175, 530.6040649, -740.4013672, 740.4013672
4: -604.0916138, 407.2537537, -604.0916138, 407.2537537, -1011.3452148, 1011.3452148

Time for backsubstitution: 1.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800662
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800662
time: 0.67 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.32
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800662
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.32
Output dim: 4, lower bound: -808.1800662, upper bound: 808.1800662
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953864
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944980
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1946924, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945059
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944996
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945629
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945087
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944993
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944996
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953604
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944928
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945004
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953924
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953604
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944991
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945004
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945382
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944946
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944997
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945629
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944993
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1954045, upper bound: 808.1944997
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945705, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1951762
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945125, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945381, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945117
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1953272, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1954001
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945107, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944958, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944970, upper bound: 808.1954001
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945107, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944952, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945424
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945485
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1948173, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1948173
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953871
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945424, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945374
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945107
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944952
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1954001, upper bound: 808.1944970
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945705
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945107
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944958
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1954001, upper bound: 808.1944970
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953272
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1953877
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945381
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945125
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944952
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1951762, upper bound: 808.1944970
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945705
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1945125
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944958
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944997, upper bound: 808.1954045
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944993, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945220, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945629, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944997, upper bound: 808.1954045
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944946, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945220, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945382, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945004, upper bound: 808.1946924
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944991, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1953604, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1953924, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945004, upper bound: 808.1946924
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944928, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1953604, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944928, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1954045
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944993, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945087, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945629, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944996, upper bound: 808.1954045
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1945059, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1946924
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944980, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1953864, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1946924
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.32
Output dim: 4, lower bound: -808.1944895, upper bound: 808.1944895

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.44 + 416.87 = 420.31 seconds
