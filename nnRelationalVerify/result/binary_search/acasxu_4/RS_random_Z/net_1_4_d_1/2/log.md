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
execution time: IAR + LP analysis = 1.78 + 0.93 = 2.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0562259, upper bound: 0.0562259


# Binary Search by BASE starts (time budget: 1197.30 seconds, max iter: 100)

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
Binary search time: 48.86 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.003663635035536572


# Relational Split (RS_random_Z) starts
Time budget: 1148.44 seconds

## Binary search (step 0) starts
Candidate diff: 0.1018318


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555177, upper bound: 0.0557260
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557260, upper bound: 0.0555177
time: 0.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0555177, upper bound: 0.0557260
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0557260, upper bound: 0.0555177

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553778, upper bound: 0.0556971
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553778, upper bound: 0.0556957
time: 0.32 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555988, upper bound: 0.0553905
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555640, upper bound: 0.0552738
time: 0.33 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0553778, upper bound: 0.0556971
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0553778, upper bound: 0.0556957
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0555988, upper bound: 0.0553905
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0555640, upper bound: 0.0552738

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0556971
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553772, upper bound: 0.0554517
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554180, upper bound: 0.0556941
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553837, upper bound: 0.0556957
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552814, upper bound: 0.0548218
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553787, upper bound: 0.0548576
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555592, upper bound: 0.0552697
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555079, upper bound: 0.0552190
time: 0.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0556971
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0553772, upper bound: 0.0554517
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0554180, upper bound: 0.0556941
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0553837, upper bound: 0.0556957
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0552814, upper bound: 0.0548218
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0553787, upper bound: 0.0548576
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0555592, upper bound: 0.0552697
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -0.0555079, upper bound: 0.0552190

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552350, upper bound: 0.0555406
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552060, upper bound: 0.0555699
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552138, upper bound: 0.0553637
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553534, upper bound: 0.0554267
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548250, upper bound: 0.0554375
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554177, upper bound: 0.0556941
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553040, upper bound: 0.0554664
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553834, upper bound: 0.0556957
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0547934
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551550, upper bound: 0.0542547
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553526, upper bound: 0.0548181
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553787, upper bound: 0.0548576
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546802, upper bound: 0.0552689
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555592, upper bound: 0.0552688
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553387, upper bound: 0.0551878
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554871, upper bound: 0.0551938
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0552350, upper bound: 0.0555406
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0552060, upper bound: 0.0555699
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0552138, upper bound: 0.0553637
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553534, upper bound: 0.0554267
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0548250, upper bound: 0.0554375
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0554177, upper bound: 0.0556941
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553040, upper bound: 0.0554664
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553834, upper bound: 0.0556957
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0547934
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0551550, upper bound: 0.0542547
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553526, upper bound: 0.0548181
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553787, upper bound: 0.0548576
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0546802, upper bound: 0.0552689
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0555592, upper bound: 0.0552688
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553387, upper bound: 0.0551878
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0554871, upper bound: 0.0551938

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548135, upper bound: 0.0549422
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548196, upper bound: 0.0552268
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546434, upper bound: 0.0553876
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552023, upper bound: 0.0555643
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548588, upper bound: 0.0553624
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552081, upper bound: 0.0553624
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553534, upper bound: 0.0554052
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553531, upper bound: 0.0553960
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544398, upper bound: 0.0550466
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543423, upper bound: 0.0549856
time: 0.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553935, upper bound: 0.0554825
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553882, upper bound: 0.0556700
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544865, upper bound: 0.0550641
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549347, upper bound: 0.0550641
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553595, upper bound: 0.0555510
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553525, upper bound: 0.0556714
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548502, upper bound: 0.0548062
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553526, upper bound: 0.0548181
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553443, upper bound: 0.0548213
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553308, upper bound: 0.0548163
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546792, upper bound: 0.0552582
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546734, upper bound: 0.0552655
time: 0.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555360, upper bound: 0.0552607
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548458, upper bound: 0.0552655
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547950, upper bound: 0.0545722
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553024, upper bound: 0.0551624
time: 0.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554742, upper bound: 0.0551681
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554640, upper bound: 0.0551677
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0548135, upper bound: 0.0549422
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0548196, upper bound: 0.0552268
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0546434, upper bound: 0.0553876
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0552023, upper bound: 0.0555643
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0548588, upper bound: 0.0553624
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0552081, upper bound: 0.0553624
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553534, upper bound: 0.0554052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553531, upper bound: 0.0553960
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0544398, upper bound: 0.0550466
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0543423, upper bound: 0.0549856
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553935, upper bound: 0.0554825
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553882, upper bound: 0.0556700
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0544865, upper bound: 0.0550641
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0549347, upper bound: 0.0550641
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553595, upper bound: 0.0555510
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553525, upper bound: 0.0556714
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0548502, upper bound: 0.0548062
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553526, upper bound: 0.0548181
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553443, upper bound: 0.0548213
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553308, upper bound: 0.0548163
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0546792, upper bound: 0.0552582
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0546734, upper bound: 0.0552655
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0555360, upper bound: 0.0552607
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0548458, upper bound: 0.0552655
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0547950, upper bound: 0.0545722
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0553024, upper bound: 0.0551624
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0554742, upper bound: 0.0551681
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 0, lower bound: -0.0554640, upper bound: 0.0551677

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547865, upper bound: 0.0550193
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548143, upper bound: 0.0552222
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542299, upper bound: 0.0549063
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542287, upper bound: 0.0549371
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0555588
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551821, upper bound: 0.0555643
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544660, upper bound: 0.0549586
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543164, upper bound: 0.0549563
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0551621
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547547, upper bound: 0.0552194
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551104, upper bound: 0.0553980
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553517, upper bound: 0.0554052
time: 0.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552114, upper bound: 0.0552312
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551353, upper bound: 0.0552531
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552209, upper bound: 0.0547805
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552374, upper bound: 0.0553656
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552170, upper bound: 0.0547419
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552353, upper bound: 0.0555434
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553571, upper bound: 0.0555488
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553454, upper bound: 0.0547306
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553519, upper bound: 0.0556714
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553515, upper bound: 0.0547298
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553458, upper bound: 0.0548181
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549422, upper bound: 0.0542672
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553381, upper bound: 0.0548157
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548776, upper bound: 0.0542534
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552962, upper bound: 0.0547923
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548538, upper bound: 0.0542829
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542783, upper bound: 0.0548551
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542023, upper bound: 0.0548551
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545998, upper bound: 0.0552408
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546529, upper bound: 0.0552323
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552489, upper bound: 0.0548517
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551619, upper bound: 0.0548517
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547639, upper bound: 0.0552376
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548006, upper bound: 0.0552431
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548810, upper bound: 0.0547421
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548231, upper bound: 0.0542756
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550128, upper bound: 0.0547600
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542871
time: 0.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553667, upper bound: 0.0551436
time: 0.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548281, upper bound: 0.0545722
time: 0.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0547865, upper bound: 0.0550193
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0548143, upper bound: 0.0552222
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0542299, upper bound: 0.0549063
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0542287, upper bound: 0.0549371
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0555588
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0551821, upper bound: 0.0555643
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0544660, upper bound: 0.0549586
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0543164, upper bound: 0.0549563
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0551621
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0547547, upper bound: 0.0552194
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0551104, upper bound: 0.0553980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0553517, upper bound: 0.0554052
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0552114, upper bound: 0.0552312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0551353, upper bound: 0.0552531
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0552209, upper bound: 0.0547805
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0552374, upper bound: 0.0553656
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0552170, upper bound: 0.0547419
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0552353, upper bound: 0.0555434
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0553571, upper bound: 0.0555488
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0553454, upper bound: 0.0547306
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0553519, upper bound: 0.0556714
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0553515, upper bound: 0.0547298
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0553458, upper bound: 0.0548181
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0549422, upper bound: 0.0542672
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0553381, upper bound: 0.0548157
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0548776, upper bound: 0.0542534
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0552962, upper bound: 0.0547923
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0548538, upper bound: 0.0542829
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0542783, upper bound: 0.0548551
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0542023, upper bound: 0.0548551
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0545998, upper bound: 0.0552408
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0546529, upper bound: 0.0552323
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0552489, upper bound: 0.0548517
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0551619, upper bound: 0.0548517
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0547639, upper bound: 0.0552376
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0548006, upper bound: 0.0552431
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0548810, upper bound: 0.0547421
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0548231, upper bound: 0.0542756
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0550128, upper bound: 0.0547600
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542871
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0553667, upper bound: 0.0551436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.60
Output dim: 0, lower bound: -0.0548281, upper bound: 0.0545722

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551224
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552007
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547914, upper bound: 0.0553244
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547465, upper bound: 0.0549413
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547234, upper bound: 0.0554195
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551556, upper bound: 0.0555464
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544739, upper bound: 0.0541406
time: 0.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546670, upper bound: 0.0547216
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543441, upper bound: 0.0547883
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542198, upper bound: 0.0547924
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544669, upper bound: 0.0550063
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547352, upper bound: 0.0549986
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552079, upper bound: 0.0546253
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0552541
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551064, upper bound: 0.0552312
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552071, upper bound: 0.0546328
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0552531
time: 0.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551353, upper bound: 0.0552300
time: 0.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552200, upper bound: 0.0547789
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552194, upper bound: 0.0545721
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552227, upper bound: 0.0553656
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552365, upper bound: 0.0552015
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548006, upper bound: 0.0542548
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0543319
time: 0.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552074, upper bound: 0.0555434
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552350, upper bound: 0.0552419
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549542, upper bound: 0.0552504
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549542, upper bound: 0.0552372
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551967, upper bound: 0.0545721
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549331, upper bound: 0.0545753
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549548, upper bound: 0.0554159
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549548, upper bound: 0.0552583
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549562, upper bound: 0.0543152
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549562, upper bound: 0.0543139
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553190, upper bound: 0.0547911
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551596, upper bound: 0.0547800
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553182, upper bound: 0.0547970
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550494, upper bound: 0.0547906
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549428, upper bound: 0.0547923
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552962, upper bound: 0.0547648
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552168
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552194
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546253, upper bound: 0.0552079
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552304, upper bound: 0.0548338
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551337, upper bound: 0.0548326
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551295, upper bound: 0.0548168
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549366, upper bound: 0.0548097
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543676, upper bound: 0.0548199
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542906, upper bound: 0.0548285
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547789, upper bound: 0.0552200
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551549, upper bound: 0.0547719
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553610, upper bound: 0.0551432
time: 0.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551224
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552007
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547914, upper bound: 0.0553244
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547465, upper bound: 0.0549413
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547234, upper bound: 0.0554195
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0551556, upper bound: 0.0555464
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0544739, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0546670, upper bound: 0.0547216
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0543441, upper bound: 0.0547883
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0542198, upper bound: 0.0547924
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0544669, upper bound: 0.0550063
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547352, upper bound: 0.0549986
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552079, upper bound: 0.0546253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0552541
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0551064, upper bound: 0.0552312
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552071, upper bound: 0.0546328
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0552531
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0551353, upper bound: 0.0552300
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552200, upper bound: 0.0547789
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552194, upper bound: 0.0545721
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552227, upper bound: 0.0553656
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552365, upper bound: 0.0552015
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0548006, upper bound: 0.0542548
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0543319
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552074, upper bound: 0.0555434
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552350, upper bound: 0.0552419
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549542, upper bound: 0.0552504
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549542, upper bound: 0.0552372
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0551967, upper bound: 0.0545721
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549331, upper bound: 0.0545753
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549548, upper bound: 0.0554159
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549548, upper bound: 0.0552583
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549562, upper bound: 0.0543152
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549562, upper bound: 0.0543139
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0553190, upper bound: 0.0547911
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0551596, upper bound: 0.0547800
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0553182, upper bound: 0.0547970
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0550494, upper bound: 0.0547906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549428, upper bound: 0.0547923
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552962, upper bound: 0.0547648
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552168
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552194
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0546253, upper bound: 0.0552079
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0552304, upper bound: 0.0548338
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0551337, upper bound: 0.0548326
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0551295, upper bound: 0.0548168
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0549366, upper bound: 0.0548097
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0543676, upper bound: 0.0548199
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0542906, upper bound: 0.0548285
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0547789, upper bound: 0.0552200
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0551549, upper bound: 0.0547719
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.48
Output dim: 0, lower bound: -0.0553610, upper bound: 0.0551432

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547789, upper bound: 0.0543283
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551977
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542435
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0552962
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541988, upper bound: 0.0550898
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0552830
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552190
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542155
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547923, upper bound: 0.0549428
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547570, upper bound: 0.0548309
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542871, upper bound: 0.0541406
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547254, upper bound: 0.0547838
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542212
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542870, upper bound: 0.0547886
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543972, upper bound: 0.0548278
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547427, upper bound: 0.0544042
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547300, upper bound: 0.0547875
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547943, upper bound: 0.0543823
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547445, upper bound: 0.0543482
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547962, upper bound: 0.0541406
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547786, upper bound: 0.0550494
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544427
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547906, upper bound: 0.0548073
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553182
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544609
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547970, upper bound: 0.0549739
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0550857
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551297
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551066
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547974, upper bound: 0.0541406
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547922, upper bound: 0.0541406
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0550892
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0552894
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552894, upper bound: 0.0547539
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552830, upper bound: 0.0547548
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551297, upper bound: 0.0547485
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549739, upper bound: 0.0547970
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553182, upper bound: 0.0547746
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552962, upper bound: 0.0547648
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548819, upper bound: 0.0541406
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547961
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548006
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547962
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542155, upper bound: 0.0548013
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548013
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551103, upper bound: 0.0547992
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551977, upper bound: 0.0547961
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543482, upper bound: 0.0547445
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543823, upper bound: 0.0547943
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549844, upper bound: 0.0547592
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542374
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547789, upper bound: 0.0543283
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0551977
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542435
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0552962
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541988, upper bound: 0.0550898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0552830
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552190
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542155
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547923, upper bound: 0.0549428
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547570, upper bound: 0.0548309
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0542871, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547254, upper bound: 0.0547838
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542212
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0542870, upper bound: 0.0547886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0543972, upper bound: 0.0548278
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547427, upper bound: 0.0544042
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547300, upper bound: 0.0547875
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547943, upper bound: 0.0543823
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547445, upper bound: 0.0543482
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547962, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547786, upper bound: 0.0550494
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544427
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547906, upper bound: 0.0548073
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553182
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544609
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547970, upper bound: 0.0549739
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0550857
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551297
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551066
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547974, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547922, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0550892
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0552894
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0552894, upper bound: 0.0547539
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0552830, upper bound: 0.0547548
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0551297, upper bound: 0.0547485
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0549739, upper bound: 0.0547970
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0553182, upper bound: 0.0547746
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0552962, upper bound: 0.0547648
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0548819, upper bound: 0.0541406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547961
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548006
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547962
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0542155, upper bound: 0.0548013
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548013
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0551103, upper bound: 0.0547992
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0551977, upper bound: 0.0547961
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0543482, upper bound: 0.0547445
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0543823, upper bound: 0.0547943
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0549844, upper bound: 0.0547592
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.59
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542374

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545514, upper bound: 0.0549617
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545384, upper bound: 0.0549632
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0548870
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547645, upper bound: 0.0552914
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547324, upper bound: 0.0542540
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546493, upper bound: 0.0552633
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544593, upper bound: 0.0549810
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544591, upper bound: 0.0549843
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546568, upper bound: 0.0551854
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546437, upper bound: 0.0551917
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544881, upper bound: 0.0550426
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544764, upper bound: 0.0550566
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552777, upper bound: 0.0547539
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544665, upper bound: 0.0546943
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552711, upper bound: 0.0547548
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544644, upper bound: 0.0546933
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553177, upper bound: 0.0547705
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550614, upper bound: 0.0547746
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550627, upper bound: 0.0545130
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550337, upper bound: 0.0545163
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551555, upper bound: 0.0547588
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551605, upper bound: 0.0547533
time: 0.32 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0545514, upper bound: 0.0549617
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0545384, upper bound: 0.0549632
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0548870
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0547645, upper bound: 0.0552914
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0547324, upper bound: 0.0542540
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0546493, upper bound: 0.0552633
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0544593, upper bound: 0.0549810
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0544591, upper bound: 0.0549843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0546568, upper bound: 0.0551854
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0546437, upper bound: 0.0551917
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0544881, upper bound: 0.0550426
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0544764, upper bound: 0.0550566
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0552777, upper bound: 0.0547539
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0544665, upper bound: 0.0546943
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0552711, upper bound: 0.0547548
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0544644, upper bound: 0.0546933
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0553177, upper bound: 0.0547705
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0550614, upper bound: 0.0547746
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0550627, upper bound: 0.0545130
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0550337, upper bound: 0.0545163
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0551555, upper bound: 0.0547588
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.45
Output dim: 0, lower bound: -0.0551605, upper bound: 0.0547533

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540433, upper bound: 0.0543476
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540658, upper bound: 0.0543452
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545399, upper bound: 0.0551333
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544717, upper bound: 0.0551355
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539427, upper bound: 0.0542539
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539521, upper bound: 0.0542539
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539548, upper bound: 0.0542246
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539659, upper bound: 0.0542246
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552594, upper bound: 0.0546367
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541746, upper bound: 0.0547299
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551405, upper bound: 0.0545892
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551390, upper bound: 0.0546413
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0546095
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551851, upper bound: 0.0546568
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548990, upper bound: 0.0545090
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548258, upper bound: 0.0545225
time: 0.34 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0540433, upper bound: 0.0543476
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0540658, upper bound: 0.0543452
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0545399, upper bound: 0.0551333
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0544717, upper bound: 0.0551355
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0539427, upper bound: 0.0542539
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0539521, upper bound: 0.0542539
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0539548, upper bound: 0.0542246
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0539659, upper bound: 0.0542246
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0552594, upper bound: 0.0546367
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0541746, upper bound: 0.0547299
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0551405, upper bound: 0.0545892
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0551390, upper bound: 0.0546413
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0546095
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0551851, upper bound: 0.0546568
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0548990, upper bound: 0.0545090
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.65
Output dim: 0, lower bound: -0.0548258, upper bound: 0.0545225

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546958, upper bound: 0.0544303
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552491, upper bound: 0.0545273
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549565, upper bound: 0.0543604
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549133, upper bound: 0.0543753
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542539, upper bound: 0.0538993
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542539, upper bound: 0.0538931
time: 0.32 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 2.64 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.64
Output dim: 0, lower bound: -0.0546958, upper bound: 0.0544303
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 2.64
Output dim: 0, lower bound: -0.0552491, upper bound: 0.0545273
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.64
Output dim: 0, lower bound: -0.0549565, upper bound: 0.0543604
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.64
Output dim: 0, lower bound: -0.0549133, upper bound: 0.0543753
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 2.64
Output dim: 0, lower bound: -0.0542539, upper bound: 0.0538993
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 2.64
Output dim: 0, lower bound: -0.0542539, upper bound: 0.0538931

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543460, upper bound: 0.0539140
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543468, upper bound: 0.0539057
time: 0.32 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 2.65 seconds
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 12, time: 2.65
Output dim: 0, lower bound: -0.0543460, upper bound: 0.0539140
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 12, time: 2.65
Output dim: 0, lower bound: -0.0543468, upper bound: 0.0539057
Binary search (step 0): status=Status.VERIFIED, low=0.1018318, high=0.2000000, mid=0.1018318, abs_max=0.058847926557064056
rel_dist={0: [-0.05600625688426092, 0.05600625688426092]}

## Binary search (step 1) starts
Candidate diff: 0.1509159


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

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
- Time for RS candidates: 0.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.58
Output dim: 0, lower bound: -0.0561025, upper bound: 0.0561138
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.58
Output dim: 0, lower bound: -0.0561138, upper bound: 0.0561025

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0560753, upper bound: 0.0560756
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0560753, upper bound: 0.0560887
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0560753, upper bound: 0.0560753
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0560753, upper bound: 0.0560780
time: 0.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0560753, upper bound: 0.0560756
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0560753, upper bound: 0.0560887
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0560753, upper bound: 0.0560753
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0560753, upper bound: 0.0560780

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555740, upper bound: 0.0555613
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555740, upper bound: 0.0560756
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555740, upper bound: 0.0555233
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557401, upper bound: 0.0560887
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556682, upper bound: 0.0557778
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556682, upper bound: 0.0560614
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553773, upper bound: 0.0557069
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553773, upper bound: 0.0553778
time: 0.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0555740, upper bound: 0.0555613
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0555740, upper bound: 0.0560756
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0555740, upper bound: 0.0555233
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0557401, upper bound: 0.0560887
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0556682, upper bound: 0.0557778
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0556682, upper bound: 0.0560614
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0553773, upper bound: 0.0557069
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0553773, upper bound: 0.0553778

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555366, upper bound: 0.0551404
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557899, upper bound: 0.0551404
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555101, upper bound: 0.0548481
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555418, upper bound: 0.0560613
time: 0.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556893, upper bound: 0.0551201
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557853, upper bound: 0.0550522
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555210, upper bound: 0.0549314
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555654, upper bound: 0.0560344
time: 0.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555034, upper bound: 0.0556834
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555034, upper bound: 0.0557777
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550277, upper bound: 0.0557575
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553143, upper bound: 0.0556627
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552406, upper bound: 0.0554639
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551863, upper bound: 0.0555797
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549803, upper bound: 0.0549808
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549803, upper bound: 0.0549808
time: 0.29 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0555366, upper bound: 0.0551404
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0557899, upper bound: 0.0551404
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0555101, upper bound: 0.0548481
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0555418, upper bound: 0.0560613
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0556893, upper bound: 0.0551201
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0557853, upper bound: 0.0550522
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0555210, upper bound: 0.0549314
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0555654, upper bound: 0.0560344
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0555034, upper bound: 0.0556834
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0555034, upper bound: 0.0557777
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0550277, upper bound: 0.0557575
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0553143, upper bound: 0.0556627
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0552406, upper bound: 0.0554639
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0551863, upper bound: 0.0555797
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0549803, upper bound: 0.0549808
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.25
Output dim: 0, lower bound: -0.0549803, upper bound: 0.0549808

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554115, upper bound: 0.0542064
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553991, upper bound: 0.0549930
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543695, upper bound: 0.0551386
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556316, upper bound: 0.0551052
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547333, upper bound: 0.0548246
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555089, upper bound: 0.0547333
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553522, upper bound: 0.0556684
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554382, upper bound: 0.0552798
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555595, upper bound: 0.0548286
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555039, upper bound: 0.0549750
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546484, upper bound: 0.0550522
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556311, upper bound: 0.0550351
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555104, upper bound: 0.0548927
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554538, upper bound: 0.0548617
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555603, upper bound: 0.0556000
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555218, upper bound: 0.0560175
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552936, upper bound: 0.0555351
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556724, upper bound: 0.0553882
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553294, upper bound: 0.0555610
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554137, upper bound: 0.0547879
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548817, upper bound: 0.0552471
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548285, upper bound: 0.0556252
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549566, upper bound: 0.0554391
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552144, upper bound: 0.0549532
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552406, upper bound: 0.0554400
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552380, upper bound: 0.0546434
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547809, upper bound: 0.0553582
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541671, upper bound: 0.0552127
time: 0.30 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0554115, upper bound: 0.0542064
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0553991, upper bound: 0.0549930
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0543695, upper bound: 0.0551386
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0556316, upper bound: 0.0551052
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0547333, upper bound: 0.0548246
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0555089, upper bound: 0.0547333
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0553522, upper bound: 0.0556684
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0554382, upper bound: 0.0552798
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0555595, upper bound: 0.0548286
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0555039, upper bound: 0.0549750
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0546484, upper bound: 0.0550522
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0556311, upper bound: 0.0550351
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0555104, upper bound: 0.0548927
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0554538, upper bound: 0.0548617
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0555603, upper bound: 0.0556000
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0555218, upper bound: 0.0560175
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0552936, upper bound: 0.0555351
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0556724, upper bound: 0.0553882
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0553294, upper bound: 0.0555610
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0554137, upper bound: 0.0547879
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0548817, upper bound: 0.0552471
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0548285, upper bound: 0.0556252
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0549566, upper bound: 0.0554391
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0552144, upper bound: 0.0549532
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0552406, upper bound: 0.0554400
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0552380, upper bound: 0.0546434
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0547809, upper bound: 0.0553582
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.36
Output dim: 0, lower bound: -0.0541671, upper bound: 0.0552127

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553801, upper bound: 0.0541768
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553076, upper bound: 0.0541768
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542064, upper bound: 0.0549917
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552075, upper bound: 0.0549633
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555157, upper bound: 0.0548526
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551889, upper bound: 0.0549592
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548541, upper bound: 0.0547326
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554200, upper bound: 0.0547298
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552080, upper bound: 0.0547384
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552021, upper bound: 0.0555419
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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552846, upper bound: 0.0546663
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552906, upper bound: 0.0551290
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555388, upper bound: 0.0547990
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552285, upper bound: 0.0542380
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542534, upper bound: 0.0548776
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552828, upper bound: 0.0548187
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543423, upper bound: 0.0543423
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554849, upper bound: 0.0549702
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553559, upper bound: 0.0548509
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554634, upper bound: 0.0546886
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549951, upper bound: 0.0542959
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549789, upper bound: 0.0543729
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551423, upper bound: 0.0551188
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0544860
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553154, upper bound: 0.0557959
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554455, upper bound: 0.0553708
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544637, upper bound: 0.0550507
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549203, upper bound: 0.0552035
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555459, upper bound: 0.0552353
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547419, upper bound: 0.0552170
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553284, upper bound: 0.0555473
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552081, upper bound: 0.0547308
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552727, upper bound: 0.0546272
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547690, upper bound: 0.0545722
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544860, upper bound: 0.0541768
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548817, upper bound: 0.0552471
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0554836
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548128, upper bound: 0.0548669
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0552788
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547300, upper bound: 0.0552985
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552144, upper bound: 0.0549532
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550209, upper bound: 0.0544114
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551536, upper bound: 0.0553201
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552371, upper bound: 0.0554400
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551970, upper bound: 0.0545980
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552124, upper bound: 0.0545770
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542816, upper bound: 0.0548991
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547809, upper bound: 0.0553521
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0549000
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0552127
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0553801, upper bound: 0.0541768
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0553076, upper bound: 0.0541768
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0542064, upper bound: 0.0549917
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552075, upper bound: 0.0549633
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0555157, upper bound: 0.0548526
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0551889, upper bound: 0.0549592
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0548541, upper bound: 0.0547326
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0554200, upper bound: 0.0547298
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552080, upper bound: 0.0547384
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552021, upper bound: 0.0555419
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552846, upper bound: 0.0546663
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552906, upper bound: 0.0551290
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0555388, upper bound: 0.0547990
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552285, upper bound: 0.0542380
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0542534, upper bound: 0.0548776
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552828, upper bound: 0.0548187
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0543423, upper bound: 0.0543423
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0554849, upper bound: 0.0549702
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0553559, upper bound: 0.0548509
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0554634, upper bound: 0.0546886
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0549951, upper bound: 0.0542959
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0549789, upper bound: 0.0543729
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0551423, upper bound: 0.0551188
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0544860
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0553154, upper bound: 0.0557959
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0554455, upper bound: 0.0553708
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0544637, upper bound: 0.0550507
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0549203, upper bound: 0.0552035
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0555459, upper bound: 0.0552353
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0547419, upper bound: 0.0552170
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0553284, upper bound: 0.0555473
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552081, upper bound: 0.0547308
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552727, upper bound: 0.0546272
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0547690, upper bound: 0.0545722
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0544860, upper bound: 0.0541768
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0548817, upper bound: 0.0552471
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0554836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0548128, upper bound: 0.0548669
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0552788
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0547300, upper bound: 0.0552985
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552144, upper bound: 0.0549532
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0550209, upper bound: 0.0544114
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0551536, upper bound: 0.0553201
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552371, upper bound: 0.0554400
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0551970, upper bound: 0.0545980
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0552124, upper bound: 0.0545770
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0542816, upper bound: 0.0548991
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0547809, upper bound: 0.0553521
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0549000
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.51
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0552127

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551776, upper bound: 0.0541406
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551365, upper bound: 0.0541406
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0548791
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551632, upper bound: 0.0549314
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546194, upper bound: 0.0544269
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553521, upper bound: 0.0547809
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545700, upper bound: 0.0548678
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551367, upper bound: 0.0548168
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552770, upper bound: 0.0545721
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547935, upper bound: 0.0545721
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551971, upper bound: 0.0547370
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552079, upper bound: 0.0546253
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551754, upper bound: 0.0555419
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0552541
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548835, upper bound: 0.0541406
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548835, upper bound: 0.0542561
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548810, upper bound: 0.0547421
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548231, upper bound: 0.0542756
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544185
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552925, upper bound: 0.0547300
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550987, upper bound: 0.0541988
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552755, upper bound: 0.0548013
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551357, upper bound: 0.0547888
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554528, upper bound: 0.0549456
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551900, upper bound: 0.0546343
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552200, upper bound: 0.0547789
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552531, upper bound: 0.0547984
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552194, upper bound: 0.0545721
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553586, upper bound: 0.0546102
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552074, upper bound: 0.0555459
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551549, upper bound: 0.0547719
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550591, upper bound: 0.0550427
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0542756
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549203, upper bound: 0.0551677
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545529, upper bound: 0.0549489
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544609, upper bound: 0.0541406
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547970
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543319, upper bound: 0.0547961
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542548, upper bound: 0.0548006
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546343, upper bound: 0.0551900
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549450, upper bound: 0.0552326
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0545721
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547519, upper bound: 0.0545760
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545363, upper bound: 0.0541406
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548665, upper bound: 0.0542136
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0550848
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548058, upper bound: 0.0541406
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542407, upper bound: 0.0549777
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0554821
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0547838
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552881
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547300, upper bound: 0.0548284
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550494, upper bound: 0.0547906
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543823, upper bound: 0.0547962
time: 0.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551279, upper bound: 0.0552905
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0547935
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552124, upper bound: 0.0554090
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552093, upper bound: 0.0554083
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547974, upper bound: 0.0541457
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547922, upper bound: 0.0541678
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548008, upper bound: 0.0541406
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548008, upper bound: 0.0541406
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551376
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551365
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551776
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551776, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551365, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551696, upper bound: 0.0548791
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551632, upper bound: 0.0549314
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0546194, upper bound: 0.0544269
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0553521, upper bound: 0.0547809
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0545700, upper bound: 0.0548678
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551367, upper bound: 0.0548168
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552770, upper bound: 0.0545721
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547935, upper bound: 0.0545721
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551971, upper bound: 0.0547370
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552079, upper bound: 0.0546253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551754, upper bound: 0.0555419
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0552541
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548835, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548835, upper bound: 0.0542561
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548810, upper bound: 0.0547421
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548231, upper bound: 0.0542756
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544185
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552925, upper bound: 0.0547300
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0550987, upper bound: 0.0541988
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552755, upper bound: 0.0548013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551357, upper bound: 0.0547888
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0554528, upper bound: 0.0549456
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551900, upper bound: 0.0546343
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552200, upper bound: 0.0547789
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552531, upper bound: 0.0547984
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552194, upper bound: 0.0545721
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0553586, upper bound: 0.0546102
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552074, upper bound: 0.0555459
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551549, upper bound: 0.0547719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0550591, upper bound: 0.0550427
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0542756
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0549203, upper bound: 0.0551677
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0545529, upper bound: 0.0549489
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0544609, upper bound: 0.0541406
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547970
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0543319, upper bound: 0.0547961
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0542548, upper bound: 0.0548006
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0546343, upper bound: 0.0551900
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0549450, upper bound: 0.0552326
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0545721
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547519, upper bound: 0.0545760
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0545363, upper bound: 0.0541406
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548665, upper bound: 0.0542136
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0550848
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548058, upper bound: 0.0541406
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0542407, upper bound: 0.0549777
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0554821
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0547838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552881
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547300, upper bound: 0.0548284
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0550494, upper bound: 0.0547906
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0543823, upper bound: 0.0547962
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0551279, upper bound: 0.0552905
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0547935
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552124, upper bound: 0.0554090
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0552093, upper bound: 0.0554083
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547974, upper bound: 0.0541457
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547922, upper bound: 0.0541678
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548008, upper bound: 0.0541406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0548008, upper bound: 0.0541406
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547485, upper bound: 0.0551376
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551365
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.53
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551776

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542331, upper bound: 0.0548026
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551103, upper bound: 0.0547992
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542847, upper bound: 0.0548478
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0547984
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0547539
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551298, upper bound: 0.0547485
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545854, upper bound: 0.0541406
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548750, upper bound: 0.0541406
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547809, upper bound: 0.0541489
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547789, upper bound: 0.0543283
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542155
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0553237
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547179, upper bound: 0.0548953
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547923, upper bound: 0.0549428
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547570, upper bound: 0.0548309
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547875, upper bound: 0.0547300
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552843, upper bound: 0.0547237
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542212, upper bound: 0.0548013
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552406, upper bound: 0.0547961
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0547548
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549827, upper bound: 0.0547895
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543050, upper bound: 0.0544537
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547943, upper bound: 0.0543823
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547445, upper bound: 0.0543482
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0543972
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547886, upper bound: 0.0542870
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547962, upper bound: 0.0541406
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549258, upper bound: 0.0542023
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548952, upper bound: 0.0542024
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544609
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547592, upper bound: 0.0549846
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542000, upper bound: 0.0549096
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549739, upper bound: 0.0547970
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547746
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544537, upper bound: 0.0543050
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551000
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541988, upper bound: 0.0550898
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553262
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547600, upper bound: 0.0550076
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542023, upper bound: 0.0549258
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552843
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542690, upper bound: 0.0548231
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0548810
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0550857
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0550892
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546156, upper bound: 0.0541435
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.35 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0551278, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0542331, upper bound: 0.0548026
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0551103, upper bound: 0.0547992
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0542847, upper bound: 0.0548478
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0547984
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0547539
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0551298, upper bound: 0.0547485
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0545854, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0548750, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547809, upper bound: 0.0541489
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547789, upper bound: 0.0543283
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0541406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0548013, upper bound: 0.0542155
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0553237
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547179, upper bound: 0.0548953
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547923, upper bound: 0.0549428
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547570, upper bound: 0.0548309
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547875, upper bound: 0.0547300
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0552843, upper bound: 0.0547237
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0542212, upper bound: 0.0548013
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0552406, upper bound: 0.0547961
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0547548
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0549827, upper bound: 0.0547895
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0550517, upper bound: 0.0543165
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0543050, upper bound: 0.0544537
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547943, upper bound: 0.0543823
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547445, upper bound: 0.0543482
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0543972
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547886, upper bound: 0.0542870
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547962, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0549258, upper bound: 0.0542023
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0548952, upper bound: 0.0542024
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544609
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547592, upper bound: 0.0549846
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0542000, upper bound: 0.0549096
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0549739, upper bound: 0.0547970
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547746
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0544537, upper bound: 0.0543050
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551000
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0541988, upper bound: 0.0550898
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553262
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547600, upper bound: 0.0550076
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0542023, upper bound: 0.0549258
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552843
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0542690, upper bound: 0.0548231
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0548810
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0550857
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0550892
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0546156, upper bound: 0.0541435
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.69
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552018, upper bound: 0.0546013
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552016, upper bound: 0.0546409
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545163, upper bound: 0.0550935
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545130, upper bound: 0.0551102
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550698, upper bound: 0.0544591
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550677, upper bound: 0.0544593
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551064, upper bound: 0.0546476
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551070, upper bound: 0.0546709
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0546742
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541119, upper bound: 0.0543539
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541294, upper bound: 0.0543524
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0546961
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546413, upper bound: 0.0551995
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545892, upper bound: 0.0551999
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547764, upper bound: 0.0542979
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546558, upper bound: 0.0552199
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546761, upper bound: 0.0552467
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546795, upper bound: 0.0552399
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546970, upper bound: 0.0552910
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547168, upper bound: 0.0552440
time: 0.34 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 2.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0552018, upper bound: 0.0546013
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0552016, upper bound: 0.0546409
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0545163, upper bound: 0.0550935
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0545130, upper bound: 0.0551102
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0550698, upper bound: 0.0544591
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0550677, upper bound: 0.0544593
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0551064, upper bound: 0.0546476
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0551070, upper bound: 0.0546709
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0547421, upper bound: 0.0546742
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0553262, upper bound: 0.0546742
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0541119, upper bound: 0.0543539
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0541294, upper bound: 0.0543524
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0547256, upper bound: 0.0547191
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0546961
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0546413, upper bound: 0.0551995
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0545892, upper bound: 0.0551999
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0547764, upper bound: 0.0542979
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0546558, upper bound: 0.0552199
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0546761, upper bound: 0.0552467
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0546795, upper bound: 0.0552399
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0546970, upper bound: 0.0552910
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 2.88
Output dim: 0, lower bound: -0.0547168, upper bound: 0.0552440

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549950, upper bound: 0.0543626
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549739, upper bound: 0.0543716
time: 0.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542657, upper bound: 0.0538913
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542657, upper bound: 0.0538707
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544480, upper bound: 0.0540592
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544488, upper bound: 0.0540334
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552066, upper bound: 0.0545763
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552065, upper bound: 0.0545855
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545806, upper bound: 0.0540267
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546413, upper bound: 0.0551482
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543686, upper bound: 0.0549775
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543622, upper bound: 0.0549992
time: 0.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545520, upper bound: 0.0552196
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545355, upper bound: 0.0548590
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539094, upper bound: 0.0544360
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539987, upper bound: 0.0544360
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0552399
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537113, upper bound: 0.0543402
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540591, upper bound: 0.0543396
time: 0.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546520, upper bound: 0.0552440
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
time: 0.35 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 2.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0549950, upper bound: 0.0543626
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0549739, upper bound: 0.0543716
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0542657, upper bound: 0.0538913
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0542657, upper bound: 0.0538707
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0544480, upper bound: 0.0540592
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0544488, upper bound: 0.0540334
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0552066, upper bound: 0.0545763
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0552065, upper bound: 0.0545855
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0545806, upper bound: 0.0540267
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0546413, upper bound: 0.0551482
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0543686, upper bound: 0.0549775
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0543622, upper bound: 0.0549992
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0545520, upper bound: 0.0552196
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0545355, upper bound: 0.0548590
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0539094, upper bound: 0.0544360
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0539987, upper bound: 0.0544360
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0545933, upper bound: 0.0552399
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0537113, upper bound: 0.0543402
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0540591, upper bound: 0.0543396
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0546520, upper bound: 0.0552440
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 2.93
Output dim: 0, lower bound: -0.0540848, upper bound: 0.0540848

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542039, upper bound: 0.0539395
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542039, upper bound: 0.0538975
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542399, upper bound: 0.0539362
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542416, upper bound: 0.0538978
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544422, upper bound: 0.0550890
time: 0.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544361, upper bound: 0.0550881
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542089, upper bound: 0.0550209
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537937, upper bound: 0.0549951
time: 0.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 3

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540293, upper bound: 0.0543087
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540528, upper bound: 0.0542783
time: 0.35 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 3.74 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0542039, upper bound: 0.0539395
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0542039, upper bound: 0.0538975
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0542399, upper bound: 0.0539362
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0542416, upper bound: 0.0538978
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0544422, upper bound: 0.0550890
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0544361, upper bound: 0.0550881
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0542089, upper bound: 0.0550209
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0537937, upper bound: 0.0549951
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0540293, upper bound: 0.0543087
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.74
Output dim: 0, lower bound: -0.0540528, upper bound: 0.0542783
Binary search (step 1): status=Status.VERIFIED, low=0.1509159, high=0.2000000, mid=0.1509159, abs_max=0.058847926557064056
rel_dist={0: [-0.056152448148144754, 0.056152448148144796]}

## Binary search (step 2) starts
Candidate diff: 0.1754579


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556561, upper bound: 0.0560377
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556561, upper bound: 0.0556561
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.0556561, upper bound: 0.0560377
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -0.0556561, upper bound: 0.0556561

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554140, upper bound: 0.0557476
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554814, upper bound: 0.0555210
time: 0.27 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554960, upper bound: 0.0555528
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0559471, upper bound: 0.0554960
time: 0.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0554140, upper bound: 0.0557476
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0554814, upper bound: 0.0555210
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0554960, upper bound: 0.0555528
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0559471, upper bound: 0.0554960

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0557152
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0557069
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554280, upper bound: 0.0554023
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554470, upper bound: 0.0554369
time: 0.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555553, upper bound: 0.0553790
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556653, upper bound: 0.0551270
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0550725
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556665, upper bound: 0.0551142
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0557152
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0553748, upper bound: 0.0557069
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0554280, upper bound: 0.0554023
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0554470, upper bound: 0.0554369
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0555553, upper bound: 0.0553790
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0556653, upper bound: 0.0551270
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0550725
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -0.0556665, upper bound: 0.0551142

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553708, upper bound: 0.0557013
time: 0.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553736, upper bound: 0.0557152
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553272, upper bound: 0.0554705
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553985, upper bound: 0.0557000
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552890, upper bound: 0.0551707
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552707, upper bound: 0.0552624
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552932, upper bound: 0.0552851
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548540, upper bound: 0.0552699
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550157, upper bound: 0.0552953
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550426, upper bound: 0.0550106
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0551052
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0550351
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550106, upper bound: 0.0550050
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551197, upper bound: 0.0550001
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556316, upper bound: 0.0550889
time: 0.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0550280
time: 0.30 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553708, upper bound: 0.0557013
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553736, upper bound: 0.0557152
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553272, upper bound: 0.0554705
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0553985, upper bound: 0.0557000
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0552890, upper bound: 0.0551707
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0552707, upper bound: 0.0552624
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0552932, upper bound: 0.0552851
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0548540, upper bound: 0.0552699
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0550157, upper bound: 0.0552953
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0550426, upper bound: 0.0550106
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0551052
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0550351
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0550106, upper bound: 0.0550050
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0551197, upper bound: 0.0550001
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0556316, upper bound: 0.0550889
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0550280

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552308, upper bound: 0.0548556
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552055, upper bound: 0.0555741
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553183, upper bound: 0.0555571
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553673, upper bound: 0.0557090
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551873, upper bound: 0.0553387
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546484, upper bound: 0.0553144
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552456, upper bound: 0.0554828
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552499, upper bound: 0.0555669
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552634, upper bound: 0.0551428
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552450, upper bound: 0.0549518
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552458, upper bound: 0.0552369
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551879, upper bound: 0.0552161
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552675, upper bound: 0.0552626
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552797, upper bound: 0.0552315
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546047, upper bound: 0.0552475
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548339, upper bound: 0.0552373
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549808, upper bound: 0.0551848
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549803, upper bound: 0.0552533
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555157, upper bound: 0.0548846
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0549399
time: 0.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552308, upper bound: 0.0548556
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552055, upper bound: 0.0555741
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0553183, upper bound: 0.0555571
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0553673, upper bound: 0.0557090
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0551873, upper bound: 0.0553387
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0546484, upper bound: 0.0553144
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552456, upper bound: 0.0554828
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552499, upper bound: 0.0555669
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552634, upper bound: 0.0551428
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552450, upper bound: 0.0549518
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552458, upper bound: 0.0552369
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0551879, upper bound: 0.0552161
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552675, upper bound: 0.0552626
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0552797, upper bound: 0.0552315
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0546047, upper bound: 0.0552475
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0548339, upper bound: 0.0552373
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0549808, upper bound: 0.0551848
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0549803, upper bound: 0.0552533
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0555157, upper bound: 0.0548846
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.53
Output dim: 0, lower bound: -0.0548286, upper bound: 0.0549399

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0548281
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552050, upper bound: 0.0548254
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547914, upper bound: 0.0553542
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547465, upper bound: 0.0549437
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552924, upper bound: 0.0555218
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552938, upper bound: 0.0555469
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553284, upper bound: 0.0555473
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553435, upper bound: 0.0556848
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542947, upper bound: 0.0548535
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547688, upper bound: 0.0549026
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546208, upper bound: 0.0552869
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552774
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552419, upper bound: 0.0548235
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552371, upper bound: 0.0554400
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552252, upper bound: 0.0554451
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552082, upper bound: 0.0555493
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545778, upper bound: 0.0550115
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552531, upper bound: 0.0551353
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551949, upper bound: 0.0549335
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552194, upper bound: 0.0547547
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546514, upper bound: 0.0552335
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552458, upper bound: 0.0551279
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545980, upper bound: 0.0551970
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551621, upper bound: 0.0550517
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0552539
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552445, upper bound: 0.0546687
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552788, upper bound: 0.0552280
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552788, upper bound: 0.0549860
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552211
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552226
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544282, upper bound: 0.0548248
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541671, upper bound: 0.0548248
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548248, upper bound: 0.0541671
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548163, upper bound: 0.0549892
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549532, upper bound: 0.0549641
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549562, upper bound: 0.0552290
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555157, upper bound: 0.0548846
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549749, upper bound: 0.0542938
time: 0.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0548281
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552050, upper bound: 0.0548254
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0547914, upper bound: 0.0553542
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0547465, upper bound: 0.0549437
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552924, upper bound: 0.0555218
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552938, upper bound: 0.0555469
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0553284, upper bound: 0.0555473
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0553435, upper bound: 0.0556848
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0542947, upper bound: 0.0548535
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0547688, upper bound: 0.0549026
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0546208, upper bound: 0.0552869
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552774
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552419, upper bound: 0.0548235
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552371, upper bound: 0.0554400
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552252, upper bound: 0.0554451
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552082, upper bound: 0.0555493
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0545778, upper bound: 0.0550115
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552531, upper bound: 0.0551353
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0551949, upper bound: 0.0549335
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552194, upper bound: 0.0547547
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0546514, upper bound: 0.0552335
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552458, upper bound: 0.0551279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0545980, upper bound: 0.0551970
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0551621, upper bound: 0.0550517
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552666, upper bound: 0.0552539
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552445, upper bound: 0.0546687
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552788, upper bound: 0.0552280
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0552788, upper bound: 0.0549860
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552211
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0545770, upper bound: 0.0552226
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0544282, upper bound: 0.0548248
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0541671, upper bound: 0.0548248
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0548248, upper bound: 0.0541671
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0548163, upper bound: 0.0549892
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0549532, upper bound: 0.0549641
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0549562, upper bound: 0.0552290
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0555157, upper bound: 0.0548846
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.45
Output dim: 0, lower bound: -0.0549749, upper bound: 0.0542938

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547809, upper bound: 0.0541489
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547789, upper bound: 0.0544179
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0548915
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547914, upper bound: 0.0553481
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544199, upper bound: 0.0550507
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549203, upper bound: 0.0551677
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543861, upper bound: 0.0550512
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549211, upper bound: 0.0551866
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551848, upper bound: 0.0554182
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547234, upper bound: 0.0554195
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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552021, upper bound: 0.0555137
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551556, upper bound: 0.0555582
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552702
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546189, upper bound: 0.0552845
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552726
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552770
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548117, upper bound: 0.0544175
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548003, upper bound: 0.0543756
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552124, upper bound: 0.0554090
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552093, upper bound: 0.0554083
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547786, upper bound: 0.0551348
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551348
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551775
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552300, upper bound: 0.0551353
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552531, upper bound: 0.0547984
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545753, upper bound: 0.0549331
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0546656
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547924, upper bound: 0.0542198
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547883, upper bound: 0.0543441
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542546, upper bound: 0.0548376
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0548376
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552312, upper bound: 0.0551064
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541678, upper bound: 0.0547922
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541457, upper bound: 0.0547974
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0550517
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551549, upper bound: 0.0547719
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552419, upper bound: 0.0552350
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552015, upper bound: 0.0552365
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0541663
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548301, upper bound: 0.0542534
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548688, upper bound: 0.0547854
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549889, upper bound: 0.0548145
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552541, upper bound: 0.0548951
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552145, upper bound: 0.0546456
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552168
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
time: 0.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552194
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548006, upper bound: 0.0541406
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547976, upper bound: 0.0549772
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0548081
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553524, upper bound: 0.0548018
time: 0.30 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547809, upper bound: 0.0541489
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547789, upper bound: 0.0544179
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0548915
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547914, upper bound: 0.0553481
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0544199, upper bound: 0.0550507
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0549203, upper bound: 0.0551677
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0543861, upper bound: 0.0550512
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0549211, upper bound: 0.0551866
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0551848, upper bound: 0.0554182
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547234, upper bound: 0.0554195
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552021, upper bound: 0.0555137
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0551556, upper bound: 0.0555582
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552702
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0546189, upper bound: 0.0552845
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552726
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552770
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0548117, upper bound: 0.0544175
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0548003, upper bound: 0.0543756
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552124, upper bound: 0.0554090
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552093, upper bound: 0.0554083
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547786, upper bound: 0.0551348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551348
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551775
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552300, upper bound: 0.0551353
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552531, upper bound: 0.0547984
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545753, upper bound: 0.0549331
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0546656
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547924, upper bound: 0.0542198
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547883, upper bound: 0.0543441
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0542546, upper bound: 0.0548376
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0541766, upper bound: 0.0548376
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552312, upper bound: 0.0551064
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0541678, upper bound: 0.0547922
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0541457, upper bound: 0.0547974
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0550517
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0551549, upper bound: 0.0547719
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552419, upper bound: 0.0552350
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552015, upper bound: 0.0552365
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0541663
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0548301, upper bound: 0.0542534
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0548688, upper bound: 0.0547854
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0549889, upper bound: 0.0548145
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552541, upper bound: 0.0548951
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0552145, upper bound: 0.0546456
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552168
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0552194
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0548006, upper bound: 0.0541406
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547976, upper bound: 0.0549772
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0548081
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 2.44
Output dim: 0, lower bound: -0.0553524, upper bound: 0.0548018

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542435
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0553237
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547592, upper bound: 0.0549846
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542000, upper bound: 0.0549096
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547600, upper bound: 0.0550076
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542023, upper bound: 0.0549258
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544537, upper bound: 0.0543050
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551000
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541988, upper bound: 0.0550898
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547895, upper bound: 0.0549827
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553262
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552843
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548651
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548058
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542075, upper bound: 0.0548835
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548835
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548665
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0545363
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548750
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0545854
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0550857
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0550892
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544609
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547875, upper bound: 0.0547300
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544042, upper bound: 0.0547427
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0543972
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547886, upper bound: 0.0542870
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543902, upper bound: 0.0541406
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547632, upper bound: 0.0542561
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547838, upper bound: 0.0547254
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542871
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549739, upper bound: 0.0547970
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548073, upper bound: 0.0547906
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548418, upper bound: 0.0544759
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548478, upper bound: 0.0542847
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548006, upper bound: 0.0541406
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548026, upper bound: 0.0542331
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547961
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548006
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547962
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547746
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550494, upper bound: 0.0547786
time: 0.33 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542435
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547648, upper bound: 0.0553237
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547592, upper bound: 0.0549846
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0542000, upper bound: 0.0549096
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547600, upper bound: 0.0550076
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0542023, upper bound: 0.0549258
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0544537, upper bound: 0.0543050
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547888, upper bound: 0.0551000
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0543165, upper bound: 0.0550517
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541988, upper bound: 0.0550898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547895, upper bound: 0.0549827
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547961, upper bound: 0.0552406
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553262
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547237, upper bound: 0.0552843
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548651
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548058
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0542075, upper bound: 0.0548835
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548835
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548665
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0545363
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548750
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0545854
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0550857
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547984, upper bound: 0.0551025
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0550892
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547992, upper bound: 0.0551103
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553340
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553282
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0544609
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0551278
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547875, upper bound: 0.0547300
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0544042, upper bound: 0.0547427
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0548278, upper bound: 0.0543972
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547886, upper bound: 0.0542870
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0543902, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547632, upper bound: 0.0542561
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0547838, upper bound: 0.0547254
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0542871
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0549739, upper bound: 0.0547970
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0548073, upper bound: 0.0547906
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0548418, upper bound: 0.0544759
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0548478, upper bound: 0.0542847
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0548006, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0548026, upper bound: 0.0542331
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547961
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0548006
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0541406
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0541406, upper bound: 0.0547962
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0553340, upper bound: 0.0547746
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 2.58
Output dim: 0, lower bound: -0.0550494, upper bound: 0.0547786

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0553237
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547764, upper bound: 0.0542979
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546558, upper bound: 0.0552199
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547324, upper bound: 0.0542540
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546493, upper bound: 0.0553065
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0552843
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545287, upper bound: 0.0551025
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545214, upper bound: 0.0551175
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544881, upper bound: 0.0551075
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544764, upper bound: 0.0551210
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543524, upper bound: 0.0541294
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543539, upper bound: 0.0541119
time: 0.32 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0546906, upper bound: 0.0553237
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0547029, upper bound: 0.0547366
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0547764, upper bound: 0.0542979
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0546558, upper bound: 0.0552199
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0547324, upper bound: 0.0542540
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0546493, upper bound: 0.0553065
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0546336, upper bound: 0.0552843
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0546462, upper bound: 0.0548031
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0545287, upper bound: 0.0551025
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0545214, upper bound: 0.0551175
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0544881, upper bound: 0.0551075
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0544764, upper bound: 0.0551210
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0543524, upper bound: 0.0541294
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.00
Output dim: 0, lower bound: -0.0543539, upper bound: 0.0541119

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540478, upper bound: 0.0543472
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540907, upper bound: 0.0543443
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545464, upper bound: 0.0550895
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544899, upper bound: 0.0550898
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545489, upper bound: 0.0553065
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544759, upper bound: 0.0547136
time: 0.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 6
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542733, upper bound: 0.0550519
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542671, upper bound: 0.0550263
time: 0.33 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 3.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.04
Output dim: 0, lower bound: -0.0540478, upper bound: 0.0543472
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.04
Output dim: 0, lower bound: -0.0540907, upper bound: 0.0543443
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.04
Output dim: 0, lower bound: -0.0545464, upper bound: 0.0550895
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.04
Output dim: 0, lower bound: -0.0544899, upper bound: 0.0550898
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 3.04
Output dim: 0, lower bound: -0.0545489, upper bound: 0.0553065
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.04
Output dim: 0, lower bound: -0.0544759, upper bound: 0.0547136
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 3.04
Output dim: 0, lower bound: -0.0542733, upper bound: 0.0550519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 3.04
Output dim: 0, lower bound: -0.0542671, upper bound: 0.0550263

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 10
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 2
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539209, upper bound: 0.0544396
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0539276, upper bound: 0.0544390
time: 0.34 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.08
Output dim: 0, lower bound: -0.0539209, upper bound: 0.0544396
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.08
Output dim: 0, lower bound: -0.0539276, upper bound: 0.0544390
Binary search (step 2): status=Status.VERIFIED, low=0.1754579, high=0.2000000, mid=0.1754579, abs_max=0.058847926557064056
rel_dist={0: [-0.05620897342554189, 0.056208973425541875]}

## Binary search (step 3) starts
Candidate diff: 0.1877290


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561544, upper bound: 0.0561544
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561544, upper bound: 0.0561553
time: 0.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 0.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.0561544, upper bound: 0.0561544
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 0.62
Output dim: 0, lower bound: -0.0561544, upper bound: 0.0561553

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561240, upper bound: 0.0561197
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561238, upper bound: 0.0561231
time: 0.30 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552027, upper bound: 0.0558759
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558764, upper bound: 0.0557568
time: 0.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0561240, upper bound: 0.0561197
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0561238, upper bound: 0.0561231
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0552027, upper bound: 0.0558759
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.35
Output dim: 0, lower bound: -0.0558764, upper bound: 0.0557568

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561205, upper bound: 0.0555613
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555740, upper bound: 0.0561144
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557278, upper bound: 0.0558414
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558415, upper bound: 0.0550522
time: 0.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551270, upper bound: 0.0556653
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0550725
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558414, upper bound: 0.0557278
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558357, upper bound: 0.0555458
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0561205, upper bound: 0.0555613
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0555740, upper bound: 0.0561144
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0557278, upper bound: 0.0558414
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0558415, upper bound: 0.0550522
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0551270, upper bound: 0.0556653
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0550725
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0558414, upper bound: 0.0557278
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -0.0558357, upper bound: 0.0555458

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0561078, upper bound: 0.0554911
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557835, upper bound: 0.0555406
time: 0.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554327, upper bound: 0.0548756
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554398, upper bound: 0.0560650
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549076, upper bound: 0.0548286
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555367, upper bound: 0.0557278
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542184, upper bound: 0.0548543
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553659, upper bound: 0.0549076
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550106, upper bound: 0.0555192
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550106, upper bound: 0.0550157
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550702, upper bound: 0.0550477
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550702, upper bound: 0.0550480
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0555260
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0550889
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0558357, upper bound: 0.0551370
time: 0.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551404, upper bound: 0.0555458
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0561078, upper bound: 0.0554911
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0557835, upper bound: 0.0555406
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0554327, upper bound: 0.0548756
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0554398, upper bound: 0.0560650
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0549076, upper bound: 0.0548286
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0555367, upper bound: 0.0557278
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0542184, upper bound: 0.0548543
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0553659, upper bound: 0.0549076
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0550106, upper bound: 0.0555192
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0550106, upper bound: 0.0550157
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0550702, upper bound: 0.0550477
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0550702, upper bound: 0.0550480
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0555260
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0550258, upper bound: 0.0550889
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0558357, upper bound: 0.0551370
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0551404, upper bound: 0.0555458

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548060, upper bound: 0.0554031
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556758, upper bound: 0.0553525
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556680, upper bound: 0.0552799
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556650, upper bound: 0.0554311
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549922, upper bound: 0.0542184
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549922, upper bound: 0.0543981
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552282, upper bound: 0.0555593
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553202, upper bound: 0.0551547
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548846, upper bound: 0.0555157
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553356, upper bound: 0.0550823
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552778, upper bound: 0.0548817
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543461, upper bound: 0.0545804
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548573, upper bound: 0.0551686
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548181, upper bound: 0.0553920
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552792, upper bound: 0.0552777
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550722, upper bound: 0.0555260
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0557140, upper bound: 0.0549882
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542184, upper bound: 0.0549922
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551104, upper bound: 0.0553520
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0543695
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 2.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0548060, upper bound: 0.0554031
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0556758, upper bound: 0.0553525
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0556680, upper bound: 0.0552799
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0556650, upper bound: 0.0554311
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0549922, upper bound: 0.0542184
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0549922, upper bound: 0.0543981
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0552282, upper bound: 0.0555593
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0553202, upper bound: 0.0551547
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0548846, upper bound: 0.0555157
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0553356, upper bound: 0.0550823
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0552778, upper bound: 0.0548817
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0543461, upper bound: 0.0545804
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0548573, upper bound: 0.0551686
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0548181, upper bound: 0.0553920
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0552792, upper bound: 0.0552777
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0550722, upper bound: 0.0555260
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0557140, upper bound: 0.0549882
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0542184, upper bound: 0.0549922
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0551104, upper bound: 0.0553520
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 2.58
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0543695

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547298, upper bound: 0.0554031
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548048, upper bound: 0.0553552
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555493, upper bound: 0.0551567
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554307, upper bound: 0.0552096
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553076, upper bound: 0.0541768
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553099, upper bound: 0.0548204
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546488, upper bound: 0.0554124
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554802, upper bound: 0.0553600
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0555593
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552280, upper bound: 0.0552788
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547949, upper bound: 0.0545722
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552906, upper bound: 0.0551290
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548024, upper bound: 0.0553585
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548082, upper bound: 0.0548248
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553063, upper bound: 0.0548846
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551610, upper bound: 0.0550564
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552778, upper bound: 0.0548817
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0544860
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551166
time: 0.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0551202
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553678
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549475, upper bound: 0.0551968
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552106, upper bound: 0.0549803
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549248, upper bound: 0.0553356
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548233, upper bound: 0.0553970
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550474, upper bound: 0.0549850
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555110, upper bound: 0.0548657
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549790, upper bound: 0.0552953
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550369, upper bound: 0.0544966
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0547298, upper bound: 0.0554031
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0548048, upper bound: 0.0553552
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0555493, upper bound: 0.0551567
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0554307, upper bound: 0.0552096
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0553076, upper bound: 0.0541768
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0553099, upper bound: 0.0548204
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0546488, upper bound: 0.0554124
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0554802, upper bound: 0.0553600
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0555593
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0552280, upper bound: 0.0552788
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0547949, upper bound: 0.0545722
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0552906, upper bound: 0.0551290
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0548024, upper bound: 0.0553585
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0548082, upper bound: 0.0548248
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0553063, upper bound: 0.0548846
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0551610, upper bound: 0.0550564
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0552778, upper bound: 0.0548817
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0544860
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0548326, upper bound: 0.0551166
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0548338, upper bound: 0.0551202
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0547911, upper bound: 0.0553678
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0549475, upper bound: 0.0551968
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0552106, upper bound: 0.0549803
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0549248, upper bound: 0.0553356
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0548233, upper bound: 0.0553970
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0550474, upper bound: 0.0549850
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0555110, upper bound: 0.0548657
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0549790, upper bound: 0.0552953
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 2.59
Output dim: 0, lower bound: -0.0550369, upper bound: 0.0544966

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543139, upper bound: 0.0550096
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543139, upper bound: 0.0549760
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545721, upper bound: 0.0545721
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546456, upper bound: 0.0552145
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551792, upper bound: 0.0541406
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553282, upper bound: 0.0547539
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551717, upper bound: 0.0548008
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551230, upper bound: 0.0548008
time: 0.32 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0541768
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551668, upper bound: 0.0541768
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541815, upper bound: 0.0545763
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551867, upper bound: 0.0548191
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0549654
time: 0.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0549629
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551632, upper bound: 0.0549314
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551386, upper bound: 0.0549244
time: 0.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547914, upper bound: 0.0553481
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547465, upper bound: 0.0549413
time: 0.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546945, upper bound: 0.0545871
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552019, upper bound: 0.0552541
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545926, upper bound: 0.0547066
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552905, upper bound: 0.0551279
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547786, upper bound: 0.0550494
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547746, upper bound: 0.0553341
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553063, upper bound: 0.0548833
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550859, upper bound: 0.0548846
time: 0.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551610, upper bound: 0.0548989
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550591, upper bound: 0.0550427
time: 0.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541768, upper bound: 0.0548817
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550044, upper bound: 0.0548652
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542420, upper bound: 0.0549313
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547800, upper bound: 0.0551596
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547548, upper bound: 0.0553281
time: 0.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547539, upper bound: 0.0553295
time: 0.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547865, upper bound: 0.0550190
time: 0.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542287, upper bound: 0.0549371
time: 0.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549955, upper bound: 0.0548157
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541663, upper bound: 0.0548177
time: 0.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548989, upper bound: 0.0551610
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548833, upper bound: 0.0553063
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542371, upper bound: 0.0551527
time: 0.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547938, upper bound: 0.0553592
time: 0.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543281, upper bound: 0.0542144
time: 0.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553481, upper bound: 0.0547914
time: 0.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479
1: -0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910
2: -0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850
3: -0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681
4: -0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295

Time for backsubstitution: 2.03 seconds
Binary search (step 3): status=Status.UNKNOWN, low=0.1754579, high=0.1877290, mid=0.1877290, abs_max=0.058847926557064056
rel_dist={0: [-0.05622070219853963, 0.05622070219853964]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.17545794394862924
execution time: 1149.85 seconds
