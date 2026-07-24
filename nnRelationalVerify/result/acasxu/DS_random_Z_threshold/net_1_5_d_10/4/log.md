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
execution time: IAR + RelationalAnalysis = 0.92 + 2.37 = 3.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -60.2372775, upper bound: 60.2372775

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2314466, upper bound: 60.2314466
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2314466, upper bound: 60.2314466
time: 0.99 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.00 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 4, lower bound: -60.2314466, upper bound: 60.2314466
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.00
Output dim: 4, lower bound: -60.2314466, upper bound: 60.2314466

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203170, upper bound: 60.2203170
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2203170, upper bound: 60.2309852
time: 0.83 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2223587
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2314239
time: 0.84 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.70 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 4, lower bound: -60.2203170, upper bound: 60.2203170
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 4, lower bound: -60.2203170, upper bound: 60.2309852
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2223587
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.70
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2314239

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2157422, upper bound: 60.2157422
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2157422, upper bound: 60.2157422
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132199, upper bound: 60.2132199
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2132199, upper bound: 60.2150660
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215817, upper bound: 60.2216797
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215817, upper bound: 60.2215817
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2314239
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2300969
time: 0.90 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.78 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -60.2157422, upper bound: 60.2157422
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -60.2157422, upper bound: 60.2157422
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -60.2132199, upper bound: 60.2132199
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -60.2132199, upper bound: 60.2150660
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -60.2215817, upper bound: 60.2216797
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -60.2215817, upper bound: 60.2215817
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2314239
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2300969

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2138749, upper bound: 60.2138749
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2138749, upper bound: 60.2138749
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2133500, upper bound: 60.2133500
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2133500, upper bound: 60.2133500
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125590, upper bound: 60.2125590
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125590, upper bound: 60.2125590
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2144310
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2150378
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1853457, upper bound: 60.1853457
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1853457, upper bound: 60.1853457
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198233, upper bound: 60.2198233
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198233, upper bound: 60.2198233
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2222480
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2314239
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2224031
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2299581
time: 1.00 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.80 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2138749, upper bound: 60.2138749
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2138749, upper bound: 60.2138749
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2133500, upper bound: 60.2133500
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2133500, upper bound: 60.2133500
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2125590, upper bound: 60.2125590
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2125590, upper bound: 60.2125590
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2144310
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2150378
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.1853457, upper bound: 60.1853457
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.1853457, upper bound: 60.1853457
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2198233, upper bound: 60.2198233
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2198233, upper bound: 60.2198233
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2222480
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2314239
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2224031
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.80
Output dim: 4, lower bound: -60.2222480, upper bound: 60.2299581

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1861885, upper bound: 60.1861814
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1861814, upper bound: 60.1861814
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2137025, upper bound: 60.2137025
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2137025, upper bound: 60.2137025
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1842416, upper bound: 60.1842416
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1842416, upper bound: 60.1842416
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1842416, upper bound: 60.1842416
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1842416, upper bound: 60.1842416
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125590, upper bound: 60.2125590
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125590, upper bound: 60.2125590
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2144310
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2150378
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193078, upper bound: 60.2193078
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2193078, upper bound: 60.2193078
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2199613, upper bound: 60.2198233
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2198233, upper bound: 60.2198233
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215817, upper bound: 60.2215817
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215817, upper bound: 60.2215817
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202907, upper bound: 60.2202907
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202907, upper bound: 60.2309633
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1933438, upper bound: 60.1934827
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1933438, upper bound: 60.1934827
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164771, upper bound: 60.2164771
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164771, upper bound: 60.2247821
time: 0.74 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.1861885, upper bound: 60.1861814
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.1861814, upper bound: 60.1861814
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2137025, upper bound: 60.2137025
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2137025, upper bound: 60.2137025
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.1842416, upper bound: 60.1842416
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.1842416, upper bound: 60.1842416
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.1842416, upper bound: 60.1842416
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.1842416, upper bound: 60.1842416
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2125590, upper bound: 60.2125590
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2125590, upper bound: 60.2125590
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2144310
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2150378
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2193078, upper bound: 60.2193078
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2193078, upper bound: 60.2193078
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2199613, upper bound: 60.2198233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2198233, upper bound: 60.2198233
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2215817, upper bound: 60.2215817
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2215817, upper bound: 60.2215817
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2202907, upper bound: 60.2202907
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2202907, upper bound: 60.2309633
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.1933438, upper bound: 60.1934827
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.1933438, upper bound: 60.1934827
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2164771, upper bound: 60.2164771
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.56
Output dim: 4, lower bound: -60.2164771, upper bound: 60.2247821

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127757, upper bound: 60.2103821
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2144310
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2150378
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2120902, upper bound: 60.2120902
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2120902, upper bound: 60.2120902
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170679, upper bound: 60.2170679
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170679, upper bound: 60.2170679
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2175379, upper bound: 60.2175379
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2175379, upper bound: 60.2175379
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1834930, upper bound: 60.1834229
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1834229, upper bound: 60.1834229
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215183, upper bound: 60.2215183
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2215183, upper bound: 60.2215183
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210649, upper bound: 60.2210649
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210649, upper bound: 60.2210649
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1920094, upper bound: 60.1920434
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1920094, upper bound: 60.1920601
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172238, upper bound: 60.2276664
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2172238, upper bound: 60.2276664
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164771, upper bound: 60.2164771
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2164771, upper bound: 60.2164771
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2237602
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2230476
time: 0.94 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2127757, upper bound: 60.2103821
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2144310
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2150378
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2120902, upper bound: 60.2120902
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2120902, upper bound: 60.2120902
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2170679, upper bound: 60.2170679
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2170679, upper bound: 60.2170679
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2175379, upper bound: 60.2175379
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2175379, upper bound: 60.2175379
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.1834930, upper bound: 60.1834229
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.1834229, upper bound: 60.1834229
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2215183, upper bound: 60.2215183
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2215183, upper bound: 60.2215183
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2210649, upper bound: 60.2210649
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2210649, upper bound: 60.2210649
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.1920094, upper bound: 60.1920434
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.1920094, upper bound: 60.1920601
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2172238, upper bound: 60.2276664
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2172238, upper bound: 60.2276664
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2164771, upper bound: 60.2164771
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2164771, upper bound: 60.2164771
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2237602
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.85
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2230476

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2107716, upper bound: 60.2083907
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2098419, upper bound: 60.2083907
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126472, upper bound: 60.2126472
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126472, upper bound: 60.2126472
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125154, upper bound: 60.2125154
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125154, upper bound: 60.2125154
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2144310
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2133522
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2129003
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2119135
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1842117, upper bound: 60.1842117
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1842117, upper bound: 60.1842117
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2120902, upper bound: 60.2120902
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2120902, upper bound: 60.2120902
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1893094, upper bound: 60.1893094
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1893094, upper bound: 60.1893094
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170679, upper bound: 60.2170679
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2170679, upper bound: 60.2170679
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1834229, upper bound: 60.1834229
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1834229, upper bound: 60.1834229
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2171487, upper bound: 60.2171487
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2171487, upper bound: 60.2171487
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210649, upper bound: 60.2210649
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2210649, upper bound: 60.2210649
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2187220, upper bound: 60.2187220
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2187220, upper bound: 60.2187220
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2168055, upper bound: 60.2274045
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2168055, upper bound: 60.2168055
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1917769, upper bound: 60.1917769
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1917769, upper bound: 60.1932714
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2136924, upper bound: 60.2136924
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2136924, upper bound: 60.2235720
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2147483
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2230476
time: 0.98 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2107716, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2098419, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2126472, upper bound: 60.2126472
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2126472, upper bound: 60.2126472
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2125154, upper bound: 60.2125154
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2125154, upper bound: 60.2125154
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2144310
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2133522
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2129003
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2119135
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.1842117, upper bound: 60.1842117
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.1842117, upper bound: 60.1842117
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2120902, upper bound: 60.2120902
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2120902, upper bound: 60.2120902
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.1893094, upper bound: 60.1893094
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.1893094, upper bound: 60.1893094
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2170679, upper bound: 60.2170679
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2170679, upper bound: 60.2170679
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.1834229, upper bound: 60.1834229
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.1834229, upper bound: 60.1834229
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2171487, upper bound: 60.2171487
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2171487, upper bound: 60.2171487
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2210649, upper bound: 60.2210649
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2210649, upper bound: 60.2210649
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2187220, upper bound: 60.2187220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2187220, upper bound: 60.2187220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2168055, upper bound: 60.2274045
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2168055, upper bound: 60.2168055
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.1917769, upper bound: 60.1917769
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.1917769, upper bound: 60.1932714
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2136924, upper bound: 60.2136924
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2136924, upper bound: 60.2235720
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2147483
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.89
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2230476

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2107716, upper bound: 60.2083907
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090697, upper bound: 60.2083907
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2098419, upper bound: 60.2083907
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083869, upper bound: 60.2083869
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083869, upper bound: 60.2083869
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2077717, upper bound: 60.2077717
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2077717, upper bound: 60.2077717
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2098898, upper bound: 60.2090564
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090564, upper bound: 60.2090564
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126331, upper bound: 60.2126331
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2126331, upper bound: 60.2126331
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125154, upper bound: 60.2125154
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2125154, upper bound: 60.2125154
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113508, upper bound: 60.2113508
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113508, upper bound: 60.2113508
time: 1.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2110325
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2110325
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2120056, upper bound: 60.2120056
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2120056, upper bound: 60.2120056
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2123452
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2110325
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2133522
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2090496
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2090496
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2090496
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2090496
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2101814
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2109211
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2116703
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2112744, upper bound: 60.2112744
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2112744, upper bound: 60.2112744
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2075310, upper bound: 60.2075310
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2075310, upper bound: 60.2075310
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2145270, upper bound: 60.2145270
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2145270, upper bound: 60.2145270
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1893094, upper bound: 60.1893094
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -60.1893094, upper bound: 60.1893094
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2171487, upper bound: 60.2171487
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2171487, upper bound: 60.2171487
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2165865, upper bound: 60.2165865
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2165865, upper bound: 60.2165865
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2182586, upper bound: 60.2182586
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2182586, upper bound: 60.2182586
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197861, upper bound: 60.2197861
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2197861, upper bound: 60.2197861
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2178515, upper bound: 60.2178515
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2178515, upper bound: 60.2178515
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2209995, upper bound: 60.2209995
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2209995, upper bound: 60.2209995
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140549, upper bound: 60.2140549
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2140549, upper bound: 60.2140549
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183326, upper bound: 60.2183326
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2183326, upper bound: 60.2183326
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2187220, upper bound: 60.2187220
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2187220, upper bound: 60.2187220
time: 0.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161694, upper bound: 60.2233807
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2161694, upper bound: 60.2161694
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1294031, 37.7388992, -12.1294031, 37.7388992, -49.8683014, 49.8683014
1: -17.1822987, 39.1265984, -17.1822987, 39.1265984, -56.3088989, 56.3088989
2: -14.7555904, 43.5125732, -14.7555904, 43.5125732, -58.2681656, 58.2681656
3: -16.1523533, 55.9294815, -16.1523533, 55.9294815, -72.0818100, 72.0818100
4: -13.7831745, 51.7584686, -13.7831745, 51.7584686, -65.5416183, 65.5416183

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167074, upper bound: 60.2167074
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -60.2167074, upper bound: 60.2167074
time: 1.17 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2107716, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090697, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2098419, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083869, upper bound: 60.2083869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083869, upper bound: 60.2083869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2127831, upper bound: 60.2127831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2077717, upper bound: 60.2077717
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2077717, upper bound: 60.2077717
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2098898, upper bound: 60.2090564
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090564, upper bound: 60.2090564
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2126331, upper bound: 60.2126331
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2126331, upper bound: 60.2126331
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2083907, upper bound: 60.2083907
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090958, upper bound: 60.2090958
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2084171, upper bound: 60.2084171
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2070609, upper bound: 60.2070609
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2104215, upper bound: 60.2104215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2125154, upper bound: 60.2125154
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2125154, upper bound: 60.2125154
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2113508, upper bound: 60.2113508
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2113508, upper bound: 60.2113508
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2110287, upper bound: 60.2110287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2110325
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2110325
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2120056, upper bound: 60.2120056
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2120056, upper bound: 60.2120056
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2123452
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2110325, upper bound: 60.2110325
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2131700
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2131700, upper bound: 60.2133522
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2113049, upper bound: 60.2113049
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2131657, upper bound: 60.2131657
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2090496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2090496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2090496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2090496
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2101814
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2090496, upper bound: 60.2109211
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2103821
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2103821, upper bound: 60.2116703
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2112744, upper bound: 60.2112744
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2112744, upper bound: 60.2112744
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2075310, upper bound: 60.2075310
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2075310, upper bound: 60.2075310
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2145270, upper bound: 60.2145270
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2145270, upper bound: 60.2145270
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.1893094, upper bound: 60.1893094
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.1893094, upper bound: 60.1893094
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2171487, upper bound: 60.2171487
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2171487, upper bound: 60.2171487
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2165865, upper bound: 60.2165865
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2165865, upper bound: 60.2165865
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2182586, upper bound: 60.2182586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2182586, upper bound: 60.2182586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2197861, upper bound: 60.2197861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2197861, upper bound: 60.2197861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2178515, upper bound: 60.2178515
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2178515, upper bound: 60.2178515
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2202989, upper bound: 60.2202989
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2209995, upper bound: 60.2209995
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2209995, upper bound: 60.2209995
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2140549, upper bound: 60.2140549
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2140549, upper bound: 60.2140549
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2183326, upper bound: 60.2183326
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2183326, upper bound: 60.2183326
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2187220, upper bound: 60.2187220
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2187220, upper bound: 60.2187220
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2161694, upper bound: 60.2233807
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2161694, upper bound: 60.2161694
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2167074, upper bound: 60.2167074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.50
Output dim: 4, lower bound: -60.2167074, upper bound: 60.2167074
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 4, lower bound: -60.2153969, upper bound: 60.2153969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 4, lower bound: -60.2136924, upper bound: 60.2136924
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 4, lower bound: -60.2136924, upper bound: 60.2235720
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2147483
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 4, lower bound: -60.2147483, upper bound: 60.2230476

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.29 + 417.22 = 420.50 seconds
