## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 60.201135133499996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014)
1: (-17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989)
2: (-14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656)
3: (-16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100)
4: (-13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.65 + 2.34 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -60.2372775, upper bound: 60.2372775

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2355433, upper bound: 60.2355433
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2355433, upper bound: 60.2372756
time: 0.83 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.75 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -60.2355433, upper bound: 60.2355433
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -60.2355433, upper bound: 60.2372756

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2215309
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2215309
time: 0.72 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2304412
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2304412
time: 0.93 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.55 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2215309
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2215309
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2304412
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.55
Output dim: 4, lower bound: -60.2215309, upper bound: 60.2304412

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1881589, upper bound: 60.1881957
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1882513, upper bound: 60.1881957
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1882513, upper bound: 60.1881589
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1882513, upper bound: 60.1881589
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1881589, upper bound: 60.1882513
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1881589, upper bound: 60.1882513
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1881957, upper bound: 60.1882513
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1881957, upper bound: 60.1882513
time: 1.00 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.70 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 4, lower bound: -60.1881589, upper bound: 60.1881957
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 4, lower bound: -60.1882513, upper bound: 60.1881957
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 4, lower bound: -60.1882513, upper bound: 60.1881589
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 4, lower bound: -60.1882513, upper bound: 60.1881589
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 4, lower bound: -60.1881589, upper bound: 60.1882513
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 4, lower bound: -60.1881589, upper bound: 60.1882513
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 4, lower bound: -60.1881957, upper bound: 60.1882513
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 4, lower bound: -60.1881957, upper bound: 60.1882513

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.99 + 15.85 = 18.84 seconds
