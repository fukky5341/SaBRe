## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 60.201135133499996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014)
1: (-17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989)
2: (-14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656)
3: (-16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100)
4: (-13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183)

## BASE Result
execution time: IAR + LP analysis = 2.07 + 2.14 = 4.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -60.2373092, upper bound: 60.2373092


# Binary Search by BASE starts (time budget: 1195.79 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=65.54161834716797
rel_dist={4: [-60.23730919021928, 60.23730919021929]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=65.54161834716797
rel_dist={4: [-60.236541379106946, 60.23654137910691]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=65.54161834716797
rel_dist={4: [-60.234725823217936, 60.23472582321793]}

## Binary search (step 3) starts
Candidate diff: 0.0208333


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0208333, mid=0.0208333, abs_max=65.54161834716797
rel_dist={4: [-60.23221675720202, 60.23221675720205]}

## Binary search (step 4) starts
Candidate diff: 0.0104167


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0104167, mid=0.0104167, abs_max=65.54161834716797
rel_dist={4: [-60.23007248428422, 60.230072484284236]}

## Binary search (step 5) starts
Candidate diff: 0.0052083


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0052083, mid=0.0052083, abs_max=65.54161834716797
rel_dist={4: [-60.22892424106238, 60.2289242410624]}

## Binary search (step 6) starts
Candidate diff: 0.0026042


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0026042, mid=0.0026042, abs_max=65.54161834716797
rel_dist={4: [-60.2283423421207, 60.2283423421207]}

## Binary search (step 7) starts
Candidate diff: 0.0013021


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0013021, mid=0.0013021, abs_max=65.54161834716797
rel_dist={4: [-60.22798780668879, 60.22798780668879]}

## Binary search (step 8) starts
Candidate diff: 0.0006510


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0006510, mid=0.0006510, abs_max=65.54161834716797
rel_dist={4: [-60.2278026105641, 60.2278026105641]}

## Binary search (step 9) starts
Candidate diff: 0.0003255


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0003255, mid=0.0003255, abs_max=65.54161834716797
rel_dist={4: [-60.227707013532644, 60.22770701353265]}

## Binary search (step 10) starts
Candidate diff: 0.0001628


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0001628, mid=0.0001628, abs_max=65.54161834716797
rel_dist={4: [-60.227659215314496, 60.2276592153145]}

## Binary search (step 11) starts
Candidate diff: 0.0000814


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000814, mid=0.0000814, abs_max=65.54161834716797
rel_dist={4: [-60.22763531679102, 60.22763531679104]}

## Binary search (step 12) starts
Candidate diff: 0.0000407


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000407, mid=0.0000407, abs_max=65.54161834716797
rel_dist={4: [-60.227623368664055, 60.22762336866404]}

## Binary search (step 13) starts
Candidate diff: 0.0000203


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000203, mid=0.0000203, abs_max=65.54161834716797
rel_dist={4: [-60.227617379735584, 60.227617396736946]}

## Binary search (step 14) starts
Candidate diff: 0.0000102


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000102, mid=0.0000102, abs_max=65.54161834716797
rel_dist={4: [-60.22761441459569, 60.22761441459569]}

## Binary search (step 15) starts
Candidate diff: 0.0000051


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000051, mid=0.0000051, abs_max=65.54161834716797
rel_dist={4: [-60.227612956083206, 60.22761293402213]}

## Binary search (step 16) starts
Candidate diff: 0.0000025


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000025, mid=0.0000025, abs_max=65.54161834716797
rel_dist={4: [-60.227612161216065, 60.22761227015873]}

## Binary search (step 17) starts
Candidate diff: 0.0000013


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000013, mid=0.0000013, abs_max=65.54161834716797
rel_dist={4: [-60.227611924177594, 60.22761215221777]}

## Binary search (step 18) starts
Candidate diff: 0.0000006


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000006, mid=0.0000006, abs_max=65.54161834716797
rel_dist={4: [-60.22761173046293, 60.22761171975189]}

## Binary Search Result
Binary search time: 75.74 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1120.05 seconds

## Binary search (step 0) starts
Candidate diff: 0.1666667


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
time: 0.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.97 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.97
Output dim: 4, lower bound: -60.2308157, upper bound: 60.2308157

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2215309
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2308157
time: 0.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2215309
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2308157
time: 0.97 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.93 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2215309
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2308157
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2215309
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.93
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2308157

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2233604, upper bound: 60.2134585
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
time: 1.37 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 4, lower bound: -60.2233604, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.05
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2165794
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
time: 1.22 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2165794
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 4.44
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2165794
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
time: 0.88 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2165794
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.96
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2233604

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2165794
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2225930
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2225930
time: 0.91 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2165794
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2225930
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2134585
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 4, lower bound: -60.2134585, upper bound: 60.2225930

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2139978
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2150700
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125659, upper bound: 60.2125659
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.14 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1666667, mid=0.1666667, abs_max=65.54161834716797
rel_dist={4: [-60.23730919021928, 60.23730919021929]}

## Binary search (step 1) starts
Candidate diff: 0.0833333


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464
time: 1.00 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.17
Output dim: 4, lower bound: -60.2290464, upper bound: 60.2290464

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212166, upper bound: 60.2212166
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212166, upper bound: 60.2290294
time: 0.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212166, upper bound: 60.2212166
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2212166, upper bound: 60.2290294
time: 0.85 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 4, lower bound: -60.2212166, upper bound: 60.2212166
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 4, lower bound: -60.2212166, upper bound: 60.2290294
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 4, lower bound: -60.2212166, upper bound: 60.2212166
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.59
Output dim: 4, lower bound: -60.2212166, upper bound: 60.2290294

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2229920, upper bound: 60.2132013
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
time: 0.93 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.61 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 4, lower bound: -60.2229920, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2163497
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
time: 0.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2163497
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2163497
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
time: 0.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2163497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.32
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2229920

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.22 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2163497
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2223557
time: 1.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2223557
time: 1.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2163497
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2223557
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2132013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.29
Output dim: 4, lower bound: -60.2132013, upper bound: 60.2223557

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2138052
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2149128
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2124000, upper bound: 60.2124000
time: 1.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.18 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0833333, mid=0.0833333, abs_max=65.54161834716797
rel_dist={4: [-60.236541379106946, 60.23654137910691]}

## Binary search (step 2) starts
Candidate diff: 0.0416667


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2245495, upper bound: 60.2245495
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2245495, upper bound: 60.2245495
time: 1.12 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.47 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.47
Output dim: 4, lower bound: -60.2245495, upper bound: 60.2245495
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.47
Output dim: 4, lower bound: -60.2245495, upper bound: 60.2245495

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203221, upper bound: 60.2203221
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203221, upper bound: 60.2244949
time: 1.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203221, upper bound: 60.2203221
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203221, upper bound: 60.2244949
time: 1.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 4, lower bound: -60.2203221, upper bound: 60.2203221
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 4, lower bound: -60.2203221, upper bound: 60.2244949
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 4, lower bound: -60.2203221, upper bound: 60.2203221
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 4, lower bound: -60.2203221, upper bound: 60.2244949

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2181676, upper bound: 60.2126149
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
time: 0.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -60.2181676, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2148894
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
time: 0.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2148894
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.92
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2148894
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
time: 0.93 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 4.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2148894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 4.06
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2148894
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
time: 0.92 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2148894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.08
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
time: 1.30 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.38
Output dim: 4, lower bound: -60.2117838, upper bound: 60.2117838
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2148894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2126149
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.38
Output dim: 4, lower bound: -60.2126149, upper bound: 60.2181676
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0416667, mid=0.0416667, abs_max=65.54161834716797
rel_dist={4: [-60.234725823217936, 60.23472582321793]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1122.01 seconds
