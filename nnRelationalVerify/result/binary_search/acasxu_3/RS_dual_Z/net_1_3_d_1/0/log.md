## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 9159.82088535655


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188)
1: (-5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188)
2: (-7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891)
3: (-2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812)
4: (-8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906)

## BASE Result
execution time: IAR + LP analysis = 1.07 + 2.08 = 3.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9164.4112857, upper bound: 9164.4112857


# Binary Search by BASE starts (time budget: 1196.86 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672155]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=11463.63671875
rel_dist={0: [-9164.409543284748, 9164.409543284746]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=11463.63671875
rel_dist={0: [-9164.405028251073, 9164.405028251073]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=11463.63671875
rel_dist={0: [-9164.397946292802, 9164.397946292804]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=11463.63671875
rel_dist={0: [-9164.392410483666, 9164.39241048367]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=11463.63671875
rel_dist={0: [-9164.388981203918, 9164.38898120392]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=11463.63671875
rel_dist={0: [-9164.387029874371, 9164.387029874371]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=11463.63671875
rel_dist={0: [-9164.38604613241, 9164.386046132408]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=11463.63671875
rel_dist={0: [-9164.38555426143, 9164.38555426143]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=11463.63671875
rel_dist={0: [-9164.385308325942, 9164.385308325942]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=11463.63671875
rel_dist={0: [-9164.385185358204, 9164.385185358202]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=11463.63671875
rel_dist={0: [-9164.385123874348, 9164.385123874348]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=11463.63671875
rel_dist={0: [-9164.385093132441, 9164.385093132441]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=11463.63671875
rel_dist={0: [-9164.385077761535, 9164.385077761537]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=11463.63671875
rel_dist={0: [-9164.385070076176, 9164.385070076176]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=11463.63671875
rel_dist={0: [-9164.385066233679, 9164.385066233677]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=11463.63671875
rel_dist={0: [-9164.385064312786, 9164.385064312788]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=11463.63671875
rel_dist={0: [-9164.385063353026, 9164.385063353024]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=11463.63671875
rel_dist={0: [-9164.385064005155, 9164.385062874397]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=11463.63671875
rel_dist={0: [-9164.385064365013, 9164.385063303624]}

## Binary Search Result
Binary search time: 65.28 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1131.58 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.62 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.75 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.87 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 0): status=Status.VERIFIED, low=0.5000000, high=1.0000000, mid=0.5000000, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672155]}

## Binary search (step 1) starts
Candidate diff: 0.7500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.82
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 1): status=Status.VERIFIED, low=0.7500000, high=1.0000000, mid=0.7500000, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 2) starts
Candidate diff: 0.8750000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.64 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.66 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.68 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.99
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 1.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 1.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 2): status=Status.VERIFIED, low=0.8750000, high=1.0000000, mid=0.8750000, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 3) starts
Candidate diff: 0.9375000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.70 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.40
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 1.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 3): status=Status.VERIFIED, low=0.9375000, high=1.0000000, mid=0.9375000, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 4) starts
Candidate diff: 0.9687500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.74 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.74
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.42 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.42
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 4): status=Status.VERIFIED, low=0.9687500, high=1.0000000, mid=0.9687500, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 5) starts
Candidate diff: 0.9843750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.75
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 5): status=Status.VERIFIED, low=0.9843750, high=1.0000000, mid=0.9843750, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 6) starts
Candidate diff: 0.9921875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.79 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.77 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.99
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 6): status=Status.VERIFIED, low=0.9921875, high=1.0000000, mid=0.9921875, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 7) starts
Candidate diff: 0.9960938


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.66 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.31 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 7): status=Status.VERIFIED, low=0.9960938, high=1.0000000, mid=0.9960938, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672155]}

## Binary search (step 8) starts
Candidate diff: 0.9980469


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.82 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.50 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.62
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.82 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 8): status=Status.VERIFIED, low=0.9980469, high=1.0000000, mid=0.9980469, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 9) starts
Candidate diff: 0.9990234


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.70 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.64 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 1.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.80
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 9): status=Status.VERIFIED, low=0.9990234, high=1.0000000, mid=0.9990234, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672155]}

## Binary search (step 10) starts
Candidate diff: 0.9995117


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.59 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 1.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.96 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.02
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 10): status=Status.VERIFIED, low=0.9995117, high=1.0000000, mid=0.9995117, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 11) starts
Candidate diff: 0.9997559


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.36 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.36
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.76 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 1.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.07 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.07
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 11): status=Status.VERIFIED, low=0.9997559, high=1.0000000, mid=0.9997559, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 12) starts
Candidate diff: 0.9998779


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 1.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.91 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.91
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.91 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.92 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 12): status=Status.VERIFIED, low=0.9998779, high=1.0000000, mid=0.9998779, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 13) starts
Candidate diff: 0.9999390


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.70 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.38
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.57 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.96 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.08 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.08
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 13): status=Status.VERIFIED, low=0.9999390, high=1.0000000, mid=0.9999390, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 14) starts
Candidate diff: 0.9999695


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.81 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.77 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.44 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 14): status=Status.VERIFIED, low=0.9999695, high=1.0000000, mid=0.9999695, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 15) starts
Candidate diff: 0.9999847


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.39 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.61 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.84 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.74
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 15): status=Status.VERIFIED, low=0.9999847, high=1.0000000, mid=0.9999847, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 16) starts
Candidate diff: 0.9999924


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.63 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 16): status=Status.VERIFIED, low=0.9999924, high=1.0000000, mid=0.9999924, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 17) starts
Candidate diff: 0.9999962


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 1.10 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.67 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.67
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.81 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.78 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.81 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.77
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 17): status=Status.VERIFIED, low=0.9999962, high=1.0000000, mid=0.9999962, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 18) starts
Candidate diff: 0.9999981


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.71 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.71
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.71
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.71
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.71
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.73 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.73
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.00
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 18): status=Status.VERIFIED, low=0.9999981, high=1.0000000, mid=0.9999981, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary search (step 19) starts
Candidate diff: 0.9999990


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
time: 0.72 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -9162.6321766, upper bound: 9162.6309101
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -9162.6309101, upper bound: 9162.6321766

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426
time: 0.75 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 2.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9160.6326426, upper bound: 9160.6329170
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6325911
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9160.6322675, upper bound: 9160.6328072
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9160.6308180, upper bound: 9160.6327111
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9160.6327111, upper bound: 9160.6308180
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9160.6328072, upper bound: 9160.6322675
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9160.6325911, upper bound: 9160.6308180
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 2.75
Output dim: 0, lower bound: -9160.6329170, upper bound: 9160.6326426

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.08 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
time: 0.83 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 2.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7131552, upper bound: 9159.7143457
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133967
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7131474, upper bound: 9159.7133839
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7128177, upper bound: 9159.7133214
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7133214, upper bound: 9159.7128177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7133839, upper bound: 9159.7131474
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7133967, upper bound: 9159.7128177
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -9159.7143457, upper bound: 9159.7131552
Binary search (step 19): status=Status.VERIFIED, low=0.9999990, high=1.0000000, mid=0.9999990, abs_max=11463.63671875
rel_dist={0: [-9164.411285672151, 9164.411285672151]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.9999990463256836
execution time: 846.42 seconds
