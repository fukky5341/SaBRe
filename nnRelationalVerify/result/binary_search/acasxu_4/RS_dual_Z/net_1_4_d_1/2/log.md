## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.055158916499999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479)
1: (-0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910)
2: (-0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850)
3: (-0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681)
4: (-0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295)

## BASE Result
execution time: IAR + LP analysis = 1.95 + 0.87 = 2.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0562259, upper bound: 0.0562259


# Binary Search by BASE starts (time budget: 1197.18 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=0.058847926557064056
rel_dist={0: [-0.05599888835187894, 0.05599888835187895]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=0.058847926557064056
rel_dist={0: [-0.05565336822157154, 0.055653368221571534]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=0.058847926557064056
rel_dist={0: [-0.05542113047992016, 0.055421130479920144]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=0.058847926557064056
rel_dist={0: [-0.05528714416572913, 0.05528714416572915]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=0.058847926557064056
rel_dist={0: [-0.055212698399098425, 0.055212698398983934]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.VERIFIED, low=0.0031250, high=0.0062500, mid=0.0031250, abs_max=0.058847926557064056
rel_dist={0: [-0.055132520784768234, 0.05513252078476821]}

## Binary search (step 6) starts
Candidate diff: 0.0046875


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0031250, high=0.0046875, mid=0.0046875, abs_max=0.058847926557064056
rel_dist={0: [-0.055185278748120487, 0.05518527874783165]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0031250, high=0.0039062, mid=0.0039062, abs_max=0.058847926557064056
rel_dist={0: [-0.055165241381868235, 0.05516524138172432]}

## Binary search (step 8) starts
Candidate diff: 0.0035156


## IAR start
Binary search (step 8): status=Status.VERIFIED, low=0.0035156, high=0.0039062, mid=0.0035156, abs_max=0.058847926557064056
rel_dist={0: [-0.05515501399134123, 0.055155013991219096]}

## Binary search (step 9) starts
Candidate diff: 0.0037109


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0035156, high=0.0037109, mid=0.0037109, abs_max=0.058847926557064056
rel_dist={0: [-0.05516014090469245, 0.055160140904564114]}

## Binary search (step 10) starts
Candidate diff: 0.0036133


## IAR start
Binary search (step 10): status=Status.VERIFIED, low=0.0036133, high=0.0037109, mid=0.0036133, abs_max=0.058847926557064056
rel_dist={0: [-0.05515757745100269, 0.05515757745087749]}

## Binary search (step 11) starts
Candidate diff: 0.0036621


## IAR start
Binary search (step 11): status=Status.VERIFIED, low=0.0036621, high=0.0037109, mid=0.0036621, abs_max=0.058847926557064056
rel_dist={0: [-0.05515885917960294, 0.05515885917960295]}

## Binary search (step 12) starts
Candidate diff: 0.0036865


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0036621, high=0.0036865, mid=0.0036865, abs_max=0.058847926557064056
rel_dist={0: [-0.05515950004240834, 0.05515950004228079]}

## Binary search (step 13) starts
Candidate diff: 0.0036743


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0036621, high=0.0036743, mid=0.0036743, abs_max=0.058847926557064056
rel_dist={0: [-0.0551591796108497, 0.05515917961084968]}

## Binary search (step 14) starts
Candidate diff: 0.0036682


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0036621, high=0.0036682, mid=0.0036682, abs_max=0.058847926557064056
rel_dist={0: [-0.05515901939457894, 0.05515901939432502]}

## Binary search (step 15) starts
Candidate diff: 0.0036652


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0036621, high=0.0036652, mid=0.0036652, abs_max=0.058847926557064056
rel_dist={0: [-0.05515893928599323, 0.05515893928586636]}

## Binary search (step 16) starts
Candidate diff: 0.0036636


## IAR start
Binary search (step 16): status=Status.VERIFIED, low=0.0036636, high=0.0036652, mid=0.0036636, abs_max=0.058847926557064056
rel_dist={0: [-0.05515889923307673, 0.05515889923294992]}

## Binary search (step 17) starts
Candidate diff: 0.0036644


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0036636, high=0.0036644, mid=0.0036644, abs_max=0.058847926557064056
rel_dist={0: [-0.05515891925993985, 0.05515891925968616]}

## Binary Search Result
Binary search time: 50.47 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.003663635035536572


# Relational Split (RS_dual_Z) starts
Time budget: 1146.71 seconds

## Binary search (step 0) starts
Candidate diff: 0.1018318


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0559646, upper bound: 0.0559682
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0559682, upper bound: 0.0559646
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0559646, upper bound: 0.0559682
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0559682, upper bound: 0.0559646

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0555729
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557567, upper bound: 0.0559675
time: 0.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0557567
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0559645
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.56 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0555729
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0557567, upper bound: 0.0559675
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0557567
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0559645

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558850, upper bound: 0.0553177
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557628, upper bound: 0.0554600
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555336, upper bound: 0.0549383
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555854, upper bound: 0.0558859
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558859, upper bound: 0.0555854
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0555336
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554600, upper bound: 0.0557628
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0558850
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0558850, upper bound: 0.0553177
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0557628, upper bound: 0.0554600
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0555336, upper bound: 0.0549383
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0555854, upper bound: 0.0558859
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0558859, upper bound: 0.0555854
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0555336
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0554600, upper bound: 0.0557628
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.46
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0558850

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554121, upper bound: 0.0548637
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554570, upper bound: 0.0548890
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553790, upper bound: 0.0550239
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552917, upper bound: 0.0550239
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550647, upper bound: 0.0545182
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550409, upper bound: 0.0544534
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551507, upper bound: 0.0554797
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549652, upper bound: 0.0550508
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550508, upper bound: 0.0549652
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554797, upper bound: 0.0551507
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550409
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0550647
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0552917
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0553790
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0554570
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548637, upper bound: 0.0554121
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0554121, upper bound: 0.0548637
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0554570, upper bound: 0.0548890
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0553790, upper bound: 0.0550239
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0552917, upper bound: 0.0550239
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0550647, upper bound: 0.0545182
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0550409, upper bound: 0.0544534
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0551507, upper bound: 0.0554797
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0549652, upper bound: 0.0550508
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0550508, upper bound: 0.0549652
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0554797, upper bound: 0.0551507
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550409
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0550647
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0552917
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0553790
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0554570
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0548637, upper bound: 0.0554121

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553975, upper bound: 0.0548340
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552760, upper bound: 0.0542730
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554411, upper bound: 0.0548631
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552760, upper bound: 0.0548474
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553584, upper bound: 0.0549837
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552710, upper bound: 0.0549996
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552466, upper bound: 0.0549288
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552469, upper bound: 0.0549990
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551437, upper bound: 0.0551544
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551350, upper bound: 0.0554629
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554629, upper bound: 0.0551350
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551544, upper bound: 0.0551437
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549990, upper bound: 0.0552469
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0552466
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549996, upper bound: 0.0552710
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549837, upper bound: 0.0553584
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548474, upper bound: 0.0552760
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548631, upper bound: 0.0554411
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542730, upper bound: 0.0552760
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548340, upper bound: 0.0553975
time: 0.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0553975, upper bound: 0.0548340
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0552760, upper bound: 0.0542730
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0554411, upper bound: 0.0548631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0552760, upper bound: 0.0548474
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0553584, upper bound: 0.0549837
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0552710, upper bound: 0.0549996
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0552466, upper bound: 0.0549288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0552469, upper bound: 0.0549990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0551437, upper bound: 0.0551544
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0551350, upper bound: 0.0554629
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0554629, upper bound: 0.0551350
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0551544, upper bound: 0.0551437
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0549990, upper bound: 0.0552469
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0552466
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0549996, upper bound: 0.0552710
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0549837, upper bound: 0.0553584
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0548474, upper bound: 0.0552760
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0548631, upper bound: 0.0554411
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0542730, upper bound: 0.0552760
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.45
Output dim: 0, lower bound: -0.0548340, upper bound: 0.0553975

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548793, upper bound: 0.0548333
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553479, upper bound: 0.0548286
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551987, upper bound: 0.0542719
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544741, upper bound: 0.0548461
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554234, upper bound: 0.0548619
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542189, upper bound: 0.0546111
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0548448
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542961, upper bound: 0.0549836
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553063, upper bound: 0.0549181
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549954
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551949, upper bound: 0.0549590
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549288
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0549096
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549847
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0549483
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548820, upper bound: 0.0554449
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550939, upper bound: 0.0550777
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0550939
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554449, upper bound: 0.0548820
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549483, upper bound: 0.0551696
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549847, upper bound: 0.0542153
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549096, upper bound: 0.0551723
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0542153
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549590, upper bound: 0.0551949
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549954, upper bound: 0.0542153
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553063
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549836, upper bound: 0.0542961
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548448, upper bound: 0.0552177
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546111, upper bound: 0.0542189
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0554234
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548461, upper bound: 0.0544741
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542719, upper bound: 0.0551987
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0553479
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548333, upper bound: 0.0548793
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0548793, upper bound: 0.0548333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0553479, upper bound: 0.0548286
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0551987, upper bound: 0.0542719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0544741, upper bound: 0.0548461
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0554234, upper bound: 0.0548619
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0542189, upper bound: 0.0546111
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0548448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0542961, upper bound: 0.0549836
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0553063, upper bound: 0.0549181
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549954
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0551949, upper bound: 0.0549590
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0549096
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549847
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0549483
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0548820, upper bound: 0.0554449
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0550939, upper bound: 0.0550777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0550939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0554449, upper bound: 0.0548820
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0549483, upper bound: 0.0551696
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0549847, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0549096, upper bound: 0.0551723
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0549590, upper bound: 0.0551949
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0549954, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553063
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0549836, upper bound: 0.0542961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0548448, upper bound: 0.0552177
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0546111, upper bound: 0.0542189
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0554234
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0548461, upper bound: 0.0544741
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0542719, upper bound: 0.0551987
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0553479
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.58
Output dim: 0, lower bound: -0.0548333, upper bound: 0.0548793

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552490, upper bound: 0.0547601
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551380, upper bound: 0.0542334
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553190, upper bound: 0.0547911
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551867, upper bound: 0.0548191
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551202, upper bound: 0.0543559
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542680, upper bound: 0.0548423
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552304, upper bound: 0.0548338
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543205, upper bound: 0.0548763
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551337, upper bound: 0.0548326
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551416, upper bound: 0.0548765
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549026, upper bound: 0.0548652
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0549244
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543384, upper bound: 0.0544946
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553489
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547575, upper bound: 0.0544061
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554118, upper bound: 0.0548487
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553917, upper bound: 0.0548397
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544946, upper bound: 0.0543384
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549244, upper bound: 0.0551386
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548652, upper bound: 0.0549026
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548765, upper bound: 0.0551416
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548989, upper bound: 0.0551610
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549314, upper bound: 0.0551632
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548833, upper bound: 0.0552733
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548791, upper bound: 0.0551696
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543559, upper bound: 0.0551202
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548191, upper bound: 0.0551867
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0553851
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548267, upper bound: 0.0553910
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542371, upper bound: 0.0551527
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0551668
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0552490
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0552490, upper bound: 0.0547601
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0551380, upper bound: 0.0542334
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0553190, upper bound: 0.0547911
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0551867, upper bound: 0.0548191
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0551202, upper bound: 0.0543559
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0542680, upper bound: 0.0548423
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0552304, upper bound: 0.0548338
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0543205, upper bound: 0.0548763
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0551337, upper bound: 0.0548326
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0551416, upper bound: 0.0548765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0549026, upper bound: 0.0548652
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0549244
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0543384, upper bound: 0.0544946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553489
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0547575, upper bound: 0.0544061
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0554118, upper bound: 0.0548487
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0553917, upper bound: 0.0548397
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0544946, upper bound: 0.0543384
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0549244, upper bound: 0.0551386
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548652, upper bound: 0.0549026
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548765, upper bound: 0.0551416
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548989, upper bound: 0.0551610
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0549314, upper bound: 0.0551632
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548833, upper bound: 0.0552733
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548791, upper bound: 0.0551696
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0543559, upper bound: 0.0551202
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548191, upper bound: 0.0551867
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0553851
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0548267, upper bound: 0.0553910
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0542371, upper bound: 0.0551527
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0551668
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0552490
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.84
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552190, upper bound: 0.0547237
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552894, upper bound: 0.0547539
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552830, upper bound: 0.0547548
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545281, upper bound: 0.0544024
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551297, upper bound: 0.0547485
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551103, upper bound: 0.0547992
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551977, upper bound: 0.0547961
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0552962
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553182
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0547216
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553182, upper bound: 0.0547746
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543024, upper bound: 0.0541678
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552962, upper bound: 0.0547648
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551000
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548183, upper bound: 0.0542136
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548478, upper bound: 0.0542847
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551977
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548064, upper bound: 0.0542282
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548026, upper bound: 0.0542331
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551297
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544024, upper bound: 0.0545281
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0552830
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0552894
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551066
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552190
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
time: 0.32 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0552190, upper bound: 0.0547237
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0552894, upper bound: 0.0547539
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0552830, upper bound: 0.0547548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0545281, upper bound: 0.0544024
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0551297, upper bound: 0.0547485
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0551103, upper bound: 0.0547992
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0551977, upper bound: 0.0547961
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0552962
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553182
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0547216
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0553182, upper bound: 0.0547746
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0543024, upper bound: 0.0541678
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0552962, upper bound: 0.0547648
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551000
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0548183, upper bound: 0.0542136
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0548478, upper bound: 0.0542847
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551977
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0548064, upper bound: 0.0542282
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0548026, upper bound: 0.0542331
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551297
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0544024, upper bound: 0.0545281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0552830
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0552894
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551066
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552190
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.72
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551722, upper bound: 0.0546795
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551817, upper bound: 0.0546761
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547348, upper bound: 0.0546752
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552812, upper bound: 0.0546745
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552776, upper bound: 0.0546742
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551555, upper bound: 0.0547588
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551605, upper bound: 0.0547533
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0552961
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546961, upper bound: 0.0553134
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547191, upper bound: 0.0547256
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553134, upper bound: 0.0546961
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547366, upper bound: 0.0547029
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552961, upper bound: 0.0546906
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0551702
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547388, upper bound: 0.0548781
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0552776
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0547421
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0552812
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0551969
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031
time: 0.33 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0551722, upper bound: 0.0546795
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0551817, upper bound: 0.0546761
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0547348, upper bound: 0.0546752
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0552812, upper bound: 0.0546745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0552776, upper bound: 0.0546742
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0551555, upper bound: 0.0547588
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0551605, upper bound: 0.0547533
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0552961
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0546961, upper bound: 0.0553134
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0547191, upper bound: 0.0547256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0553134, upper bound: 0.0546961
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0547366, upper bound: 0.0547029
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0552961, upper bound: 0.0546906
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0551702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0547388, upper bound: 0.0548781
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0552776
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0547421
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0552812
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0551969
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.95
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551609, upper bound: 0.0545933
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547643, upper bound: 0.0546333
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551608, upper bound: 0.0546168
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551810, upper bound: 0.0546520
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552478, upper bound: 0.0546680
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551850, upper bound: 0.0546525
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552401, upper bound: 0.0546675
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548470, upper bound: 0.0547334
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551341, upper bound: 0.0546827
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546812, upper bound: 0.0552580
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546719, upper bound: 0.0551744
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546866, upper bound: 0.0552831
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546852, upper bound: 0.0551795
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0546852
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552831, upper bound: 0.0546866
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551744, upper bound: 0.0546719
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552580, upper bound: 0.0546812
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546827, upper bound: 0.0551341
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546882, upper bound: 0.0551384
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546675, upper bound: 0.0552401
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546525, upper bound: 0.0551850
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546680, upper bound: 0.0552478
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546520, upper bound: 0.0551810
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546168, upper bound: 0.0551608
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0551609
time: 0.31 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0551609, upper bound: 0.0545933
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0547643, upper bound: 0.0546333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0551608, upper bound: 0.0546168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0551810, upper bound: 0.0546520
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0552478, upper bound: 0.0546680
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0551850, upper bound: 0.0546525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0552401, upper bound: 0.0546675
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0548470, upper bound: 0.0547334
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0551341, upper bound: 0.0546827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546812, upper bound: 0.0552580
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546719, upper bound: 0.0551744
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546866, upper bound: 0.0552831
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546852, upper bound: 0.0551795
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0546852
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0552831, upper bound: 0.0546866
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0551744, upper bound: 0.0546719
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0552580, upper bound: 0.0546812
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546827, upper bound: 0.0551341
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546882, upper bound: 0.0551384
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546675, upper bound: 0.0552401
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546525, upper bound: 0.0551850
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546680, upper bound: 0.0552478
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546520, upper bound: 0.0551810
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0546168, upper bound: 0.0551608
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.28
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0551609

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548944, upper bound: 0.0537937
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0542089
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548801, upper bound: 0.0542490
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548814, upper bound: 0.0542555
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551702, upper bound: 0.0546348
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552348, upper bound: 0.0546536
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551741, upper bound: 0.0546353
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552269, upper bound: 0.0546531
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546658, upper bound: 0.0548326
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546574, upper bound: 0.0552501
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546556, upper bound: 0.0540389
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546475, upper bound: 0.0551601
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546715, upper bound: 0.0550260
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546591, upper bound: 0.0552826
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546704, upper bound: 0.0546886
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546476, upper bound: 0.0551665
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551665, upper bound: 0.0546476
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546886, upper bound: 0.0546704
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552826, upper bound: 0.0546591
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550260, upper bound: 0.0546715
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551601, upper bound: 0.0546475
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0546556
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552501, upper bound: 0.0546574
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0546658
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546531, upper bound: 0.0552269
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546353, upper bound: 0.0551741
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546536, upper bound: 0.0552348
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546348, upper bound: 0.0551702
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545987, upper bound: 0.0551548
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545739, upper bound: 0.0551548
time: 0.36 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 3.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0548944, upper bound: 0.0537937
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0542089
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0548801, upper bound: 0.0542490
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0548814, upper bound: 0.0542555
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0551702, upper bound: 0.0546348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0552348, upper bound: 0.0546536
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0551741, upper bound: 0.0546353
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0552269, upper bound: 0.0546531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546658, upper bound: 0.0548326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546574, upper bound: 0.0552501
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546556, upper bound: 0.0540389
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546475, upper bound: 0.0551601
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546715, upper bound: 0.0550260
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546591, upper bound: 0.0552826
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546704, upper bound: 0.0546886
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546476, upper bound: 0.0551665
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0551665, upper bound: 0.0546476
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546886, upper bound: 0.0546704
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0552826, upper bound: 0.0546591
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0550260, upper bound: 0.0546715
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0551601, upper bound: 0.0546475
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0546556
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0552501, upper bound: 0.0546574
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0546658
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546531, upper bound: 0.0552269
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546353, upper bound: 0.0551741
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546536, upper bound: 0.0552348
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0546348, upper bound: 0.0551702
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0545987, upper bound: 0.0551548
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.53
Output dim: 0, lower bound: -0.0545739, upper bound: 0.0551548

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549097, upper bound: 0.0537445
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549472, upper bound: 0.0542670
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549784, upper bound: 0.0542626
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549759, upper bound: 0.0543092
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549347, upper bound: 0.0537445
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549511, upper bound: 0.0542674
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549670, upper bound: 0.0542614
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549668, upper bound: 0.0543084
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549578
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542863, upper bound: 0.0549645
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542807, upper bound: 0.0549268
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548003
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543184, upper bound: 0.0549997
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542928, upper bound: 0.0550259
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542808, upper bound: 0.0549432
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548219
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548219, upper bound: 0.0537445
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549432, upper bound: 0.0542808
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550259, upper bound: 0.0542928
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549997, upper bound: 0.0543184
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548003, upper bound: 0.0537445
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549268, upper bound: 0.0542807
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549645, upper bound: 0.0542863
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549578, upper bound: 0.0543084
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549668
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542614, upper bound: 0.0549670
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542674, upper bound: 0.0549511
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549347
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543092, upper bound: 0.0549759
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542626, upper bound: 0.0549784
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542670, upper bound: 0.0549472
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549097
time: 0.35 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 3.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549097, upper bound: 0.0537445
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549472, upper bound: 0.0542670
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549784, upper bound: 0.0542626
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549759, upper bound: 0.0543092
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549347, upper bound: 0.0537445
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549511, upper bound: 0.0542674
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549670, upper bound: 0.0542614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549668, upper bound: 0.0543084
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549578
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0542863, upper bound: 0.0549645
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0542807, upper bound: 0.0549268
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548003
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0543184, upper bound: 0.0549997
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0542928, upper bound: 0.0550259
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0542808, upper bound: 0.0549432
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548219
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0548219, upper bound: 0.0537445
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549432, upper bound: 0.0542808
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0550259, upper bound: 0.0542928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549997, upper bound: 0.0543184
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0548003, upper bound: 0.0537445
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549268, upper bound: 0.0542807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549645, upper bound: 0.0542863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0549578, upper bound: 0.0543084
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549668
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0542614, upper bound: 0.0549670
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0542674, upper bound: 0.0549511
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549347
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0543092, upper bound: 0.0549759
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0542626, upper bound: 0.0549784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0542670, upper bound: 0.0549472
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 3.61
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549097
Binary search (step 0): status=Status.VERIFIED, low=0.1018318, high=0.2000000, mid=0.1018318, abs_max=0.058847926557064056
rel_dist={0: [-0.05600625688426092, 0.05600625688426092]}

## Binary search (step 1) starts
Candidate diff: 0.1509159


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561025, upper bound: 0.0561138
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561138, upper bound: 0.0561025
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0561025, upper bound: 0.0561138
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0561138, upper bound: 0.0561025

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0555729
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557567, upper bound: 0.0561138
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0557567
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0561025
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0555729
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -0.0557567, upper bound: 0.0561138
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0557567
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0561025

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0560477, upper bound: 0.0553177
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558290, upper bound: 0.0554600
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555336, upper bound: 0.0549383
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555854, upper bound: 0.0560594
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0560594, upper bound: 0.0555854
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0555336
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554600, upper bound: 0.0558290
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553177, upper bound: 0.0560477
time: 0.29 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0560477, upper bound: 0.0553177
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0558290, upper bound: 0.0554600
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0555336, upper bound: 0.0549383
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0555854, upper bound: 0.0560594
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0560594, upper bound: 0.0555854
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0555336
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0554600, upper bound: 0.0558290
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0553177, upper bound: 0.0560477

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555877, upper bound: 0.0548637
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556831, upper bound: 0.0548890
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550239
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554160, upper bound: 0.0550239
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548890, upper bound: 0.0545182
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550409, upper bound: 0.0544534
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551790, upper bound: 0.0557160
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549652, upper bound: 0.0550508
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550508, upper bound: 0.0549652
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557160, upper bound: 0.0551790
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550409
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0550647
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0554160
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0555373
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0556831
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548637, upper bound: 0.0555877
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0555877, upper bound: 0.0548637
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0556831, upper bound: 0.0548890
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550239
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0554160, upper bound: 0.0550239
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0548890, upper bound: 0.0545182
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0550409, upper bound: 0.0544534
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0551790, upper bound: 0.0557160
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0549652, upper bound: 0.0550508
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0550508, upper bound: 0.0549652
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0557160, upper bound: 0.0551790
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550409
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0550647
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0554160
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0555373
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0556831
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0548637, upper bound: 0.0555877

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555669, upper bound: 0.0548340
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553410, upper bound: 0.0542730
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556636, upper bound: 0.0548631
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553437, upper bound: 0.0548474
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553425, upper bound: 0.0549288
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553352, upper bound: 0.0549990
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551734, upper bound: 0.0551544
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551537, upper bound: 0.0556955
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556955, upper bound: 0.0551537
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551544, upper bound: 0.0551734
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549990, upper bound: 0.0553352
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0553425
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549996, upper bound: 0.0553407
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549837, upper bound: 0.0555184
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548474, upper bound: 0.0553437
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548631, upper bound: 0.0556636
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542730, upper bound: 0.0553410
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548340, upper bound: 0.0555669
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0555669, upper bound: 0.0548340
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0553410, upper bound: 0.0542730
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0556636, upper bound: 0.0548631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0553437, upper bound: 0.0548474
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0553425, upper bound: 0.0549288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0553352, upper bound: 0.0549990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0551734, upper bound: 0.0551544
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0551537, upper bound: 0.0556955
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0556955, upper bound: 0.0551537
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0551544, upper bound: 0.0551734
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0549990, upper bound: 0.0553352
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0553425
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0549996, upper bound: 0.0553407
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0549837, upper bound: 0.0555184
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0548474, upper bound: 0.0553437
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0548631, upper bound: 0.0556636
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0542730, upper bound: 0.0553410
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.59
Output dim: 0, lower bound: -0.0548340, upper bound: 0.0555669

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548793, upper bound: 0.0548333
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553899, upper bound: 0.0548286
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551987, upper bound: 0.0542719
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544741, upper bound: 0.0548461
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555229, upper bound: 0.0548619
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542189, upper bound: 0.0546111
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0548448
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549288
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0549096
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549847
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0549483
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548887, upper bound: 0.0551490
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551164, upper bound: 0.0549196
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548820, upper bound: 0.0555235
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550939, upper bound: 0.0550777
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0550939
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555235, upper bound: 0.0548820
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549196, upper bound: 0.0551164
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551490, upper bound: 0.0548887
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549483, upper bound: 0.0551696
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549847, upper bound: 0.0542153
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549096, upper bound: 0.0551723
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0542153
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549590, upper bound: 0.0551949
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549954, upper bound: 0.0542153
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553379
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549836, upper bound: 0.0542961
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548448, upper bound: 0.0552177
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546111, upper bound: 0.0542189
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0555229
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548461, upper bound: 0.0544741
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542719, upper bound: 0.0551987
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0553899
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548333, upper bound: 0.0548793
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548793, upper bound: 0.0548333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0553899, upper bound: 0.0548286
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551987, upper bound: 0.0542719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0544741, upper bound: 0.0548461
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0555229, upper bound: 0.0548619
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0542189, upper bound: 0.0546111
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0548448
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0549096
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549847
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0549483
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548887, upper bound: 0.0551490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551164, upper bound: 0.0549196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548820, upper bound: 0.0555235
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0550939, upper bound: 0.0550777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0550939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0555235, upper bound: 0.0548820
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549196, upper bound: 0.0551164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0551490, upper bound: 0.0548887
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549483, upper bound: 0.0551696
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549847, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549096, upper bound: 0.0551723
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549590, upper bound: 0.0551949
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549954, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553379
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0549836, upper bound: 0.0542961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548448, upper bound: 0.0552177
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0546111, upper bound: 0.0542189
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0555229
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548461, upper bound: 0.0544741
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0542719, upper bound: 0.0551987
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0553899
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.64
Output dim: 0, lower bound: -0.0548333, upper bound: 0.0548793

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553168, upper bound: 0.0547601
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551380, upper bound: 0.0542334
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553678, upper bound: 0.0547911
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545645, upper bound: 0.0544377
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551596, upper bound: 0.0547800
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551416, upper bound: 0.0548765
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550044, upper bound: 0.0548652
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0549244
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543384, upper bound: 0.0544946
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553686
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547575, upper bound: 0.0544061
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554888, upper bound: 0.0548487
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554790, upper bound: 0.0548397
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544946, upper bound: 0.0543384
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549244, upper bound: 0.0551386
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548652, upper bound: 0.0550044
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548765, upper bound: 0.0551416
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548989, upper bound: 0.0551610
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549314, upper bound: 0.0551632
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548833, upper bound: 0.0553063
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548791, upper bound: 0.0551696
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544377, upper bound: 0.0545645
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553678
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542334, upper bound: 0.0551380
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0553168
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0553168, upper bound: 0.0547601
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0551380, upper bound: 0.0542334
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0553678, upper bound: 0.0547911
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0545645, upper bound: 0.0544377
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0551596, upper bound: 0.0547800
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0551416, upper bound: 0.0548765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0550044, upper bound: 0.0548652
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0549244
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0543384, upper bound: 0.0544946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553686
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0547575, upper bound: 0.0544061
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0554888, upper bound: 0.0548487
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0554790, upper bound: 0.0548397
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0544946, upper bound: 0.0543384
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0549244, upper bound: 0.0551386
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0548652, upper bound: 0.0550044
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0548765, upper bound: 0.0551416
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0548989, upper bound: 0.0551610
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0549314, upper bound: 0.0551632
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0548833, upper bound: 0.0553063
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0548791, upper bound: 0.0551696
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0544377, upper bound: 0.0545645
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553678
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0542334, upper bound: 0.0551380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0553168
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.78
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552843, upper bound: 0.0547237
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0547539
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0547548
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551298, upper bound: 0.0547485
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0553237
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0547216
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547746
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543024, upper bound: 0.0541678
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553237, upper bound: 0.0547648
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551000
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548183, upper bound: 0.0542136
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548478, upper bound: 0.0542847
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548064, upper bound: 0.0542282
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548026, upper bound: 0.0542331
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551298
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553262
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552843
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
time: 0.32 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0552843, upper bound: 0.0547237
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0547539
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0547548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0551298, upper bound: 0.0547485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0553237
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0547216
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547746
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0543024, upper bound: 0.0541678
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0553237, upper bound: 0.0547648
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551000
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0548183, upper bound: 0.0542136
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0548478, upper bound: 0.0542847
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0548064, upper bound: 0.0542282
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0548026, upper bound: 0.0542331
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551298
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552843
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.78
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552399, upper bound: 0.0546795
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552467, upper bound: 0.0546761
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547348, upper bound: 0.0546752
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0546745
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0546742
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0553237
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546961, upper bound: 0.0553340
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547191, upper bound: 0.0547256
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0546961
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547366, upper bound: 0.0547029
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553237, upper bound: 0.0546906
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0552403
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547388, upper bound: 0.0548825
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0553262
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0547421
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.34 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0553282
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0552843
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031
time: 0.32 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0552399, upper bound: 0.0546795
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0552467, upper bound: 0.0546761
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0547348, upper bound: 0.0546752
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0546745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0546742
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0553237
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0546961, upper bound: 0.0553340
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0547191, upper bound: 0.0547256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0546961
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0547366, upper bound: 0.0547029
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0553237, upper bound: 0.0546906
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0552403
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0547388, upper bound: 0.0548825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0553262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0547421
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0553282
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0552843
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.97
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552399, upper bound: 0.0545933
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547643, upper bound: 0.0546333
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552467, upper bound: 0.0546168
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552440, upper bound: 0.0546520
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552910, upper bound: 0.0546680
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552634, upper bound: 0.0546525
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552886, upper bound: 0.0546675
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546812, upper bound: 0.0552867
time: 0.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546719, upper bound: 0.0552099
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546866, upper bound: 0.0552996
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546852, upper bound: 0.0552264
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552264, upper bound: 0.0546852
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552996, upper bound: 0.0546866
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553020, upper bound: 0.0546724
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548754, upper bound: 0.0546806
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546827, upper bound: 0.0552040
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546882, upper bound: 0.0552021
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546675, upper bound: 0.0552886
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546525, upper bound: 0.0552634
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546680, upper bound: 0.0552910
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546520, upper bound: 0.0552440
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546168, upper bound: 0.0552467
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0552399
time: 0.32 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 2.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0552399, upper bound: 0.0545933
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0547643, upper bound: 0.0546333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0552467, upper bound: 0.0546168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0552440, upper bound: 0.0546520
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0552910, upper bound: 0.0546680
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0552634, upper bound: 0.0546525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0552886, upper bound: 0.0546675
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546812, upper bound: 0.0552867
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546719, upper bound: 0.0552099
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546866, upper bound: 0.0552996
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546852, upper bound: 0.0552264
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0552264, upper bound: 0.0546852
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0552996, upper bound: 0.0546866
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0553020, upper bound: 0.0546724
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0548754, upper bound: 0.0546806
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546827, upper bound: 0.0552040
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546882, upper bound: 0.0552021
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546675, upper bound: 0.0552886
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546525, upper bound: 0.0552634
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546680, upper bound: 0.0552910
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546520, upper bound: 0.0552440
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0546168, upper bound: 0.0552467
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.87
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0552399

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551601, upper bound: 0.0545739
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551613, upper bound: 0.0545987
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551934, upper bound: 0.0546348
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552576, upper bound: 0.0546536
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551930, upper bound: 0.0546353
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552486, upper bound: 0.0546531
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546658, upper bound: 0.0548326
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546574, upper bound: 0.0552649
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546556, upper bound: 0.0540389
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546475, upper bound: 0.0551812
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546715, upper bound: 0.0550260
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546591, upper bound: 0.0552995
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546704, upper bound: 0.0546886
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546476, upper bound: 0.0551935
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551935, upper bound: 0.0546476
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546886, upper bound: 0.0546704
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552995, upper bound: 0.0546591
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550260, upper bound: 0.0546715
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551812, upper bound: 0.0546475
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552649, upper bound: 0.0546574
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546670, upper bound: 0.0540389
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546659, upper bound: 0.0551425
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546726, upper bound: 0.0540389
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546658, upper bound: 0.0551444
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546531, upper bound: 0.0552486
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546353, upper bound: 0.0551930
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546536, upper bound: 0.0552576
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546348, upper bound: 0.0551934
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545987, upper bound: 0.0551613
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545739, upper bound: 0.0551601
time: 0.33 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 2.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0551601, upper bound: 0.0545739
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0551613, upper bound: 0.0545987
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0551934, upper bound: 0.0546348
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0552576, upper bound: 0.0546536
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0551930, upper bound: 0.0546353
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0552486, upper bound: 0.0546531
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546658, upper bound: 0.0548326
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546574, upper bound: 0.0552649
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546556, upper bound: 0.0540389
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546475, upper bound: 0.0551812
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546715, upper bound: 0.0550260
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546591, upper bound: 0.0552995
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546704, upper bound: 0.0546886
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546476, upper bound: 0.0551935
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0551935, upper bound: 0.0546476
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546886, upper bound: 0.0546704
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0552995, upper bound: 0.0546591
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0550260, upper bound: 0.0546715
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0551812, upper bound: 0.0546475
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0552649, upper bound: 0.0546574
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546670, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546659, upper bound: 0.0551425
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546726, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546658, upper bound: 0.0551444
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546531, upper bound: 0.0552486
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546353, upper bound: 0.0551930
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546536, upper bound: 0.0552576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0546348, upper bound: 0.0551934
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0545987, upper bound: 0.0551613
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.94
Output dim: 0, lower bound: -0.0545739, upper bound: 0.0551601

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548775, upper bound: 0.0537445
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549110, upper bound: 0.0541771
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548737, upper bound: 0.0542213
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548734, upper bound: 0.0542282
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549097, upper bound: 0.0537445
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549691, upper bound: 0.0542670
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549982, upper bound: 0.0542626
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549979, upper bound: 0.0543092
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549347, upper bound: 0.0537445
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549689, upper bound: 0.0542674
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549830, upper bound: 0.0542614
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549832, upper bound: 0.0543084
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549802
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542863, upper bound: 0.0549823
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542807, upper bound: 0.0549461
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548003
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543184, upper bound: 0.0550270
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542928, upper bound: 0.0550504
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542808, upper bound: 0.0549691
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548219
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548219, upper bound: 0.0537445
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549691, upper bound: 0.0542808
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550504, upper bound: 0.0542928
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550270, upper bound: 0.0543184
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548003, upper bound: 0.0537445
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549461, upper bound: 0.0542807
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549823, upper bound: 0.0542863
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549802, upper bound: 0.0543084
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549832
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542614, upper bound: 0.0549830
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542674, upper bound: 0.0549689
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549347
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543092, upper bound: 0.0549979
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542626, upper bound: 0.0549982
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542670, upper bound: 0.0549691
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549097
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542282, upper bound: 0.0548734
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542213, upper bound: 0.0548737
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541771, upper bound: 0.0549110
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548775
time: 0.34 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 4.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0548775, upper bound: 0.0537445
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549110, upper bound: 0.0541771
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0548737, upper bound: 0.0542213
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0548734, upper bound: 0.0542282
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549097, upper bound: 0.0537445
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549691, upper bound: 0.0542670
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549982, upper bound: 0.0542626
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549979, upper bound: 0.0543092
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549347, upper bound: 0.0537445
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549689, upper bound: 0.0542674
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549830, upper bound: 0.0542614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549832, upper bound: 0.0543084
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549802
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542863, upper bound: 0.0549823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542807, upper bound: 0.0549461
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548003
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0543184, upper bound: 0.0550270
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542928, upper bound: 0.0550504
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542808, upper bound: 0.0549691
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548219
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0548219, upper bound: 0.0537445
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549691, upper bound: 0.0542808
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0550504, upper bound: 0.0542928
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0550270, upper bound: 0.0543184
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0548003, upper bound: 0.0537445
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549461, upper bound: 0.0542807
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549823, upper bound: 0.0542863
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0549802, upper bound: 0.0543084
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0543084, upper bound: 0.0549832
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542614, upper bound: 0.0549830
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542674, upper bound: 0.0549689
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549347
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0543092, upper bound: 0.0549979
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542626, upper bound: 0.0549982
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542670, upper bound: 0.0549691
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0549097
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542282, upper bound: 0.0548734
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0542213, upper bound: 0.0548737
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0541771, upper bound: 0.0549110
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 4.01
Output dim: 0, lower bound: -0.0537445, upper bound: 0.0548775
Binary search (step 1): status=Status.VERIFIED, low=0.1509159, high=0.2000000, mid=0.1509159, abs_max=0.058847926557064056
rel_dist={0: [-0.056152448148144754, 0.056152448148144796]}

## Binary search (step 2) starts
Candidate diff: 0.1754579


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561478, upper bound: 0.0561478
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561478, upper bound: 0.0561488
time: 0.29 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.0561478, upper bound: 0.0561478
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.0561478, upper bound: 0.0561488

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0555729
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0561478
time: 0.29 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561478, upper bound: 0.0557567
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0561488
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0555729
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0561478
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0561478, upper bound: 0.0557567
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -0.0555729, upper bound: 0.0561488

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0560983, upper bound: 0.0553177
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558290, upper bound: 0.0554600
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555336, upper bound: 0.0549383
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555854, upper bound: 0.0560983
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0560983, upper bound: 0.0555854
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0555336
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554600, upper bound: 0.0558290
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553177, upper bound: 0.0560983
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0560983, upper bound: 0.0553177
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0558290, upper bound: 0.0554600
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0555336, upper bound: 0.0549383
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0555854, upper bound: 0.0560983
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0560983, upper bound: 0.0555854
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0549383, upper bound: 0.0555336
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0554600, upper bound: 0.0558290
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -0.0553177, upper bound: 0.0560983

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556103, upper bound: 0.0548637
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557542, upper bound: 0.0548890
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555573, upper bound: 0.0550239
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554200, upper bound: 0.0550239
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548890, upper bound: 0.0545182
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550409, upper bound: 0.0544534
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551790, upper bound: 0.0557548
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549652, upper bound: 0.0550508
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550508, upper bound: 0.0549652
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557548, upper bound: 0.0551790
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550409
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0550647
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0554200
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0555573
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0557542
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0556103
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0556103, upper bound: 0.0548637
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0557542, upper bound: 0.0548890
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0555573, upper bound: 0.0550239
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0554200, upper bound: 0.0550239
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0548890, upper bound: 0.0545182
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0550409, upper bound: 0.0544534
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0551790, upper bound: 0.0557548
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0549652, upper bound: 0.0550508
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0550508, upper bound: 0.0549652
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0557548, upper bound: 0.0551790
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0550409
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0550647
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0554200
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0550239, upper bound: 0.0555573
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0544534, upper bound: 0.0557542
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.45
Output dim: 0, lower bound: -0.0545182, upper bound: 0.0556103

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555849, upper bound: 0.0548340
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553410, upper bound: 0.0542730
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557389, upper bound: 0.0548631
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553437, upper bound: 0.0548474
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555338, upper bound: 0.0549837
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553407, upper bound: 0.0549996
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553433, upper bound: 0.0549288
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553352, upper bound: 0.0549990
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551734, upper bound: 0.0551544
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551537, upper bound: 0.0557389
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557389, upper bound: 0.0551537
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551544, upper bound: 0.0551734
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549990, upper bound: 0.0553352
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0553433
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549996, upper bound: 0.0553407
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549837, upper bound: 0.0555338
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548474, upper bound: 0.0553437
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548631, upper bound: 0.0557389
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542730, upper bound: 0.0553410
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548340, upper bound: 0.0555849
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0555849, upper bound: 0.0548340
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0553410, upper bound: 0.0542730
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0557389, upper bound: 0.0548631
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0553437, upper bound: 0.0548474
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0555338, upper bound: 0.0549837
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0553407, upper bound: 0.0549996
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0553433, upper bound: 0.0549288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0553352, upper bound: 0.0549990
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0551734, upper bound: 0.0551544
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0551537, upper bound: 0.0557389
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0557389, upper bound: 0.0551537
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0551544, upper bound: 0.0551734
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0549990, upper bound: 0.0553352
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0553433
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0549996, upper bound: 0.0553407
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0549837, upper bound: 0.0555338
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0548474, upper bound: 0.0553437
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0548631, upper bound: 0.0557389
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0542730, upper bound: 0.0553410
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0548340, upper bound: 0.0555849

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548793, upper bound: 0.0548333
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553903, upper bound: 0.0548286
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551987, upper bound: 0.0542719
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544741, upper bound: 0.0548461
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555229, upper bound: 0.0548619
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542189, upper bound: 0.0546111
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0548448
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542961, upper bound: 0.0549836
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553379, upper bound: 0.0549181
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549954
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551949, upper bound: 0.0549590
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549288
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0549096
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549847
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0549483
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548887, upper bound: 0.0551490
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551164, upper bound: 0.0549196
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548820, upper bound: 0.0555235
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550939, upper bound: 0.0550777
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0550939
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555235, upper bound: 0.0548820
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549196, upper bound: 0.0551164
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551490, upper bound: 0.0548887
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549483, upper bound: 0.0551696
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549847, upper bound: 0.0542153
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549096, upper bound: 0.0551723
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0542153
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549590, upper bound: 0.0551949
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549954, upper bound: 0.0542153
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553379
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549836, upper bound: 0.0542961
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548448, upper bound: 0.0552177
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546111, upper bound: 0.0542189
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0555229
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548461, upper bound: 0.0544741
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542719, upper bound: 0.0551987
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0553903
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548333, upper bound: 0.0548793
time: 0.30 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0548793, upper bound: 0.0548333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0553903, upper bound: 0.0548286
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0551987, upper bound: 0.0542719
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0544741, upper bound: 0.0548461
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0555229, upper bound: 0.0548619
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0542189, upper bound: 0.0546111
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0548448
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0542961, upper bound: 0.0549836
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0553379, upper bound: 0.0549181
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549954
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0551949, upper bound: 0.0549590
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549288
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0549096
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0549847
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0549483
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0548887, upper bound: 0.0551490
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0551164, upper bound: 0.0549196
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0548820, upper bound: 0.0555235
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0550939, upper bound: 0.0550777
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0550777, upper bound: 0.0550939
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0555235, upper bound: 0.0548820
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549196, upper bound: 0.0551164
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0551490, upper bound: 0.0548887
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549483, upper bound: 0.0551696
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549847, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549096, upper bound: 0.0551723
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549288, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549590, upper bound: 0.0551949
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549954, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549181, upper bound: 0.0553379
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0549836, upper bound: 0.0542961
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0548448, upper bound: 0.0552177
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0546111, upper bound: 0.0542189
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0555229
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0548461, upper bound: 0.0544741
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0542719, upper bound: 0.0551987
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0542153, upper bound: 0.0542153
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0553903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.57
Output dim: 0, lower bound: -0.0548333, upper bound: 0.0548793

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553174, upper bound: 0.0547601
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551380, upper bound: 0.0542334
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553678, upper bound: 0.0547911
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545645, upper bound: 0.0544377
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551596, upper bound: 0.0547800
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542680, upper bound: 0.0548423
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552726, upper bound: 0.0548338
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543205, upper bound: 0.0548763
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551337, upper bound: 0.0548326
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551416, upper bound: 0.0548765
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550044, upper bound: 0.0548652
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0549244
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543384, upper bound: 0.0544946
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553686
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547575, upper bound: 0.0544061
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554888, upper bound: 0.0548487
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554790, upper bound: 0.0548397
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551166
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548673, upper bound: 0.0545126
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0551202
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548355, upper bound: 0.0541766
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551337
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548763, upper bound: 0.0543205
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0552726
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548423, upper bound: 0.0542680
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544377, upper bound: 0.0545645
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553678
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542334, upper bound: 0.0551380
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0553174
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
time: 0.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0553174, upper bound: 0.0547601
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0551380, upper bound: 0.0542334
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0553678, upper bound: 0.0547911
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0545645, upper bound: 0.0544377
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0551596, upper bound: 0.0547800
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0542680, upper bound: 0.0548423
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0552726, upper bound: 0.0548338
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0543205, upper bound: 0.0548763
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0551337, upper bound: 0.0548326
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0551416, upper bound: 0.0548765
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0550044, upper bound: 0.0548652
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0549244
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0543384, upper bound: 0.0544946
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548095, upper bound: 0.0553686
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0547575, upper bound: 0.0544061
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0554888, upper bound: 0.0548487
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0554790, upper bound: 0.0548397
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551166
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548673, upper bound: 0.0545126
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0551202
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548355, upper bound: 0.0541766
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551337
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548763, upper bound: 0.0543205
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0552726
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0548423, upper bound: 0.0542680
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0544377, upper bound: 0.0545645
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553678
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0542334, upper bound: 0.0551380
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0547601, upper bound: 0.0553174
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.64
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0541766

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552843, upper bound: 0.0547237
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0547539
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0547548
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551298, upper bound: 0.0547485
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551103, upper bound: 0.0547992
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552406, upper bound: 0.0547961
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0553237
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0547216
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547746
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543024, upper bound: 0.0541678
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553237, upper bound: 0.0547648
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551298
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553262
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552843
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0552843, upper bound: 0.0547237
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0547539
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0547548
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0551298, upper bound: 0.0547485
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0551103, upper bound: 0.0547992
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0552406, upper bound: 0.0547961
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0553237
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0543703, upper bound: 0.0547216
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547746
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0543024, upper bound: 0.0541678
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0553237, upper bound: 0.0547648
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551298
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552843
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.66
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552399, upper bound: 0.0546795
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552467, upper bound: 0.0546761
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547348, upper bound: 0.0546752
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0546745
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0546742
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548825, upper bound: 0.0547388
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552403, upper bound: 0.0547001
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.34 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0553237
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546961, upper bound: 0.0553340
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547191, upper bound: 0.0547256
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0546961
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547366, upper bound: 0.0547029
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553237, upper bound: 0.0546906
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.32 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0552403
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547388, upper bound: 0.0548825
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0553262
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0547421
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.31 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0553282
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.33 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0552843
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031
time: 0.32 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0552399, upper bound: 0.0546795
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0552467, upper bound: 0.0546761
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0547348, upper bound: 0.0546752
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0546745
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0546742
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0548825, upper bound: 0.0547388
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0552403, upper bound: 0.0547001
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0553237
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0546961, upper bound: 0.0553340
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0547191, upper bound: 0.0547256
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0546961
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0547366, upper bound: 0.0547029
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0553237, upper bound: 0.0546906
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0552403
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0547388, upper bound: 0.0548825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0553262
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0546742, upper bound: 0.0547421
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0546745, upper bound: 0.0553282
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0546752, upper bound: 0.0547348
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0552843
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.96
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552399, upper bound: 0.0545933
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547643, upper bound: 0.0546333
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552467, upper bound: 0.0546168
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552440, upper bound: 0.0546520
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552910, upper bound: 0.0546680
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552634, upper bound: 0.0546525
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552886, upper bound: 0.0546675
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552021, upper bound: 0.0546882
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552040, upper bound: 0.0546827
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546806, upper bound: 0.0548754
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546724, upper bound: 0.0553020
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546866, upper bound: 0.0552996
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546852, upper bound: 0.0552264
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552264, upper bound: 0.0546852
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552996, upper bound: 0.0546866
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553020, upper bound: 0.0546724
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548754, upper bound: 0.0546806
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546827, upper bound: 0.0552040
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546882, upper bound: 0.0552021
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540713, upper bound: 0.0540713
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546682, upper bound: 0.0552823
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540713, upper bound: 0.0540713
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546687, upper bound: 0.0552870
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546168, upper bound: 0.0552467
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0552399
time: 0.34 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552399, upper bound: 0.0545933
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0547643, upper bound: 0.0546333
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552467, upper bound: 0.0546168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552440, upper bound: 0.0546520
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552910, upper bound: 0.0546680
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552634, upper bound: 0.0546525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552886, upper bound: 0.0546675
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552021, upper bound: 0.0546882
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552040, upper bound: 0.0546827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546806, upper bound: 0.0548754
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546724, upper bound: 0.0553020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546866, upper bound: 0.0552996
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546852, upper bound: 0.0552264
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552264, upper bound: 0.0546852
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0552996, upper bound: 0.0546866
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0553020, upper bound: 0.0546724
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0548754, upper bound: 0.0546806
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546827, upper bound: 0.0552040
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546882, upper bound: 0.0552021
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0540713, upper bound: 0.0540713
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546682, upper bound: 0.0552823
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0540713, upper bound: 0.0540713
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546687, upper bound: 0.0552870
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0546168, upper bound: 0.0552467
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 3.06
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0552399

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551601, upper bound: 0.0545739
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551613, upper bound: 0.0545987
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551934, upper bound: 0.0546348
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552576, upper bound: 0.0546536
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551930, upper bound: 0.0546353
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552486, upper bound: 0.0546531
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0540389
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551444, upper bound: 0.0546658
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0546726
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551425, upper bound: 0.0546659
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540389, upper bound: 0.0546670
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546574, upper bound: 0.0552649
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546475, upper bound: 0.0551812
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546715, upper bound: 0.0550260
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546591, upper bound: 0.0552995
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546704, upper bound: 0.0546886
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546476, upper bound: 0.0551935
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551935, upper bound: 0.0546476
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546886, upper bound: 0.0546704
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552995, upper bound: 0.0546591
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550260, upper bound: 0.0546715
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551812, upper bound: 0.0546475
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552649, upper bound: 0.0546574
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546670, upper bound: 0.0540389
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546659, upper bound: 0.0551425
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.19 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.1509159, high=0.1754579, mid=0.1754579, abs_max=0.058847926557064056
rel_dist={0: [-0.056208973425541896, 0.056208973425541875]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.15091589981818743
execution time: 1147.58 seconds
