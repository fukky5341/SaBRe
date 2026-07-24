## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 71.14967792064


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007)
1: (-29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329)
2: (-15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983)
3: (-33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246)
4: (-20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.82 + 1.77 = 2.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -71.1567936, upper bound: 71.1567936

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1510815, upper bound: 71.1510815
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1510815, upper bound: 71.1510815
time: 0.92 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.86 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 4, lower bound: -71.1510815, upper bound: 71.1510815
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 4, lower bound: -71.1510815, upper bound: 71.1510815

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007
1: -29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329
2: -15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983
3: -33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246
4: -20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1492728, upper bound: 71.1509498
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1509498, upper bound: 71.1489890
time: 0.71 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007
1: -29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329
2: -15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983
3: -33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246
4: -20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1510815, upper bound: 71.1508183
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1508183, upper bound: 71.1510815
time: 0.60 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 4, lower bound: -71.1492728, upper bound: 71.1509498
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 4, lower bound: -71.1509498, upper bound: 71.1489890
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 4, lower bound: -71.1510815, upper bound: 71.1508183
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 4, lower bound: -71.1508183, upper bound: 71.1510815

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007
1: -29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329
2: -15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983
3: -33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246
4: -20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1399234, upper bound: 71.1416932
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1391120, upper bound: 71.1416932
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007
1: -29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329
2: -15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983
3: -33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246
4: -20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1506224, upper bound: 71.1484476
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1493946, upper bound: 71.1482161
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007
1: -29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329
2: -15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983
3: -33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246
4: -20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1488388, upper bound: 71.1491477
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1493673, upper bound: 71.1476355
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007
1: -29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329
2: -15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983
3: -33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246
4: -20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1483985, upper bound: 71.1494796
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -71.1483985, upper bound: 71.1507534
time: 1.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.74 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.74
Output dim: 4, lower bound: -71.1399234, upper bound: 71.1416932
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.74
Output dim: 4, lower bound: -71.1391120, upper bound: 71.1416932
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 4, lower bound: -71.1506224, upper bound: 71.1484476
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.74
Output dim: 4, lower bound: -71.1493946, upper bound: 71.1482161
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.74
Output dim: 4, lower bound: -71.1488388, upper bound: 71.1491477
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.74
Output dim: 4, lower bound: -71.1493673, upper bound: 71.1476355
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.74
Output dim: 4, lower bound: -71.1483985, upper bound: 71.1494796
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 4, lower bound: -71.1483985, upper bound: 71.1507534

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007
1: -29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329
2: -15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983
3: -33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246
4: -20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1395886, upper bound: 71.1395926
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1395886, upper bound: 71.1395926
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -86.8210449, 164.3859558, -86.8210449, 164.3859558, -251.2070007, 251.2070007
1: -29.7035065, 57.3727303, -29.7035065, 57.3727303, -87.0762329, 87.0762329
2: -15.8138723, 59.4475250, -15.8138723, 59.4475250, -75.2613983, 75.2613983
3: -33.6637955, 71.3761292, -33.6637955, 71.3761292, -105.0399246, 105.0399246
4: -20.2065468, 58.7098618, -20.2065468, 58.7098618, -78.9164124, 78.9164124

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1446712, upper bound: 71.1446515
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -71.1446390, upper bound: 71.1450051
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.95 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.95
Output dim: 4, lower bound: -71.1395886, upper bound: 71.1395926
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.95
Output dim: 4, lower bound: -71.1395886, upper bound: 71.1395926
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.95
Output dim: 4, lower bound: -71.1446712, upper bound: 71.1446515
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.95
Output dim: 4, lower bound: -71.1446390, upper bound: 71.1450051

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.60 + 19.53 = 22.13 seconds
